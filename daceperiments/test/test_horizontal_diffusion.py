import dace
import numpy as np
import pytest


@pytest.fixture()
def horizontal_diffusion():
    nx, ny, nz = 10, 13, 17

    inp = np.random.uniform(size=(nx, ny, nz))
    coeff = np.random.uniform(size=(nx, ny, nz))

    lap = np.zeros_like(inp)
    lap[1:-1, 1:-1, :] = 4 * inp[1:-1, 1:-1, :] - (
        inp[2:, 1:-1, :] + inp[:-2, 1:-1, :] + inp[1:-1, 2:, :] +
        inp[1:-1, :-2, :])

    flx = np.zeros_like(inp)
    flx[:-1, :, :] = lap[1:, :, :] - lap[:-1, :, :]
    flx[:-1, :, :] = np.where(
        flx[:-1, :, :] * (inp[1:, :, :] - inp[:-1, :, :]) > 0, 0,
        flx[:-1, :, :])

    fly = np.zeros_like(inp)
    fly[:, :-1, :] = lap[:, 1:, :] - lap[:, :-1, :]
    fly[:, :-1, :] = np.where(
        fly[:, :-1, :] * (inp[:, 1:, :] - inp[:, :-1, :]) > 0, 0,
        fly[:, :-1, :])

    out = np.zeros_like(inp)
    out[1:-1, 1:-1, :] = inp[1:-1, 1:-1, :] - coeff[1:-1, 1:-1, :] * (
        flx[1:-1, 1:-1, :] - flx[:-2, 1:-1, :] + fly[1:-1, 1:-1, :] -
        fly[1:-1, :-2, :])

    return inp, coeff, out


def generate_sdfg(name='hdiff'):
    sdfg = dace.SDFG(name)

    nx = dace.symbol('nx', dtype=dace.dtypes.int32)
    ny = dace.symbol('ny', dtype=dace.dtypes.int32)
    nz = dace.symbol('nz', dtype=dace.dtypes.int32)

    inp = sdfg.add_array('inp', (nx, ny, nz), dace.dtypes.float64)
    coeff = sdfg.add_array('coeff', (nx, ny, nz), dace.dtypes.float64)
    out = sdfg.add_array('out', (nx, ny, nz), dace.dtypes.float64)

    lap = sdfg.add_transient('lap', (nx, ny, nz), dace.dtypes.float64)
    flx = sdfg.add_transient('flx', (nx, ny, nz), dace.dtypes.float64)
    fly = sdfg.add_transient('fly', (nx, ny, nz), dace.dtypes.float64)

    state = sdfg.add_state()

    inp_read = state.add_read('inp')
    coeff_read = state.add_read('coeff')
    out_write = state.add_write('out')

    # lap
    lap_map_entry, lap_map_exit = state.add_map('lap_map',
                                                ndrange=dict(i='1:nx-1',
                                                             j='1:ny-1',
                                                             k='0:nz'))

    tasklet = state.add_tasklet(
        'lap_tasklet',
        inputs={'inp_im1j', 'inp_ijm1', 'inp_ij', 'inp_ip1j', 'inp_ijp1'},
        outputs={'lap_ij'},
        code='lap_ij = 4 * inp_ij - (inp_im1j + inp_ijm1 + inp_ip1j + inp_ijp1)'
    )

    lap_map_entry.add_in_connector('IN_inp')
    state.add_edge(inp_read, None, lap_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', subset_str='0:nx, 0:ny, 0:nz'))

    lap_map_entry.add_out_connector('OUT_inp')
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_im1j',
                   dace.Memlet.simple('inp', subset_str='i-1, j, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ip1j',
                   dace.Memlet.simple('inp', subset_str='i+1, j, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ijm1',
                   dace.Memlet.simple('inp', subset_str='i, j-1, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ijp1',
                   dace.Memlet.simple('inp', subset_str='i, j+1, k'))

    lap_map_exit.add_in_connector('IN_lap')
    state.add_edge(tasklet, 'lap_ij', lap_map_exit, 'IN_lap',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))

    lap_map_exit.add_out_connector('OUT_lap')
    lap_access = state.add_access('lap')
    state.add_edge(
        lap_map_exit, 'OUT_lap', lap_access, None,
        dace.Memlet.simple('lap', subset_str='1:nx-1, 1:ny-1, 0:nz'))

    # flx
    flx_map_entry, flx_map_exit = state.add_map('flx_map',
                                                ndrange=dict(i='0:nx-1',
                                                             j='0:ny',
                                                             k='0:nz'))
    tasklet = state.add_tasklet(
        'flx_tasklet',
        inputs={'inp_ij', 'inp_ip1j', 'lap_ij', 'lap_ip1j'},
        outputs={'flx_ij'},
        code='flx_ij = lap_ip1j - lap_ij\n'
        'if flx_ij * (inp_ip1j - inp_ij) > 0: flx_ij = 0')

    flx_map_entry.add_in_connector('IN_inp')
    flx_map_entry.add_in_connector('IN_lap')
    state.add_edge(inp_read, None, flx_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', subset_str='1:nx, 0:ny, 0:nz'))
    state.add_edge(lap_access, None, flx_map_entry, 'IN_lap',
                   dace.Memlet.simple('lap', subset_str='1:nx, 0:ny, 0:nz'))

    flx_map_entry.add_out_connector('OUT_inp')
    flx_map_entry.add_out_connector('OUT_lap')
    state.add_edge(flx_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(flx_map_entry, 'OUT_inp', tasklet, 'inp_ip1j',
                   dace.Memlet.simple('inp', subset_str='i+1, j, k'))
    state.add_edge(flx_map_entry, 'OUT_lap', tasklet, 'lap_ij',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))
    state.add_edge(flx_map_entry, 'OUT_lap', tasklet, 'lap_ip1j',
                   dace.Memlet.simple('lap', subset_str='i+1, j, k'))

    flx_map_exit.add_in_connector('IN_flx')
    state.add_edge(tasklet, 'flx_ij', flx_map_exit, 'IN_flx',
                   dace.Memlet.simple('flx', subset_str='i, j, k'))

    flx_map_exit.add_out_connector('OUT_flx')
    flx_access = state.add_access('flx')
    state.add_edge(flx_map_exit, 'OUT_flx', flx_access, None,
                   dace.Memlet.simple('flx', subset_str='0:nx-1, 0:ny, 0:nz'))

    # fly
    fly_map_entry, fly_map_exit = state.add_map('fly_map',
                                                ndrange=dict(i='0:nx',
                                                             j='0:ny-1',
                                                             k='0:nz'))
    tasklet = state.add_tasklet(
        'fly_tasklet',
        inputs={'inp_ij', 'inp_ijp1', 'lap_ij', 'lap_ijp1'},
        outputs={'fly_ij'},
        code='fly_ij = lap_ijp1 - lap_ij\n'
        'if fly_ij * (inp_ijp1 - inp_ij) > 0: fly_ij = 0')

    fly_map_entry.add_in_connector('IN_inp')
    fly_map_entry.add_in_connector('IN_lap')
    state.add_edge(inp_read, None, fly_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', subset_str='1:nx, 0:ny, 0:nz'))
    state.add_edge(lap_access, None, fly_map_entry, 'IN_lap',
                   dace.Memlet.simple('lap', subset_str='1:nx, 0:ny, 0:nz'))

    fly_map_entry.add_out_connector('OUT_inp')
    fly_map_entry.add_out_connector('OUT_lap')
    state.add_edge(fly_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(fly_map_entry, 'OUT_inp', tasklet, 'inp_ijp1',
                   dace.Memlet.simple('inp', subset_str='i, j+1, k'))
    state.add_edge(fly_map_entry, 'OUT_lap', tasklet, 'lap_ij',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))
    state.add_edge(fly_map_entry, 'OUT_lap', tasklet, 'lap_ijp1',
                   dace.Memlet.simple('lap', subset_str='i, j+1, k'))

    fly_map_exit.add_in_connector('IN_fly')
    state.add_edge(tasklet, 'fly_ij', fly_map_exit, 'IN_fly',
                   dace.Memlet.simple('fly', subset_str='i, j, k'))

    fly_map_exit.add_out_connector('OUT_fly')
    fly_access = state.add_access('fly')
    state.add_edge(fly_map_exit, 'OUT_fly', fly_access, None,
                   dace.Memlet.simple('fly', subset_str='0:nx, 0:ny-1, 0:nz'))

    # out
    out_map_entry, out_map_exit = state.add_map('out_map',
                                                ndrange=dict(i='1:nx-1',
                                                             j='1:ny-1',
                                                             k='0:nz'))

    tasklet = state.add_tasklet(
        'out_tasklet',
        inputs={
            'inp_ij', 'coeff_ij', 'flx_im1j', 'flx_ij', 'fly_ijm1', 'fly_ij'
        },
        outputs={'out_ij'},
        code=
        'out_ij = inp_ij - coeff_ij * (flx_ij - flx_im1j + fly_ij - fly_ijm1)')

    out_map_entry.add_in_connector('IN_inp')
    out_map_entry.add_in_connector('IN_coeff')
    out_map_entry.add_in_connector('IN_flx')
    out_map_entry.add_in_connector('IN_fly')
    state.add_edge(inp_read, None, out_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', '1:nx, 1:ny, 0:nz'))
    state.add_edge(coeff_read, None, out_map_entry, 'IN_coeff',
                   dace.Memlet.simple('coeff', '1:nx, 1:ny, 0:nz'))
    state.add_edge(flx_access, None, out_map_entry, 'IN_flx',
                   dace.Memlet.simple('flx', '1:nx, 0:ny, 0:nz'))
    state.add_edge(fly_access, None, out_map_entry, 'IN_fly',
                   dace.Memlet.simple('fly', '0:nx, 1:ny, 0:nz'))

    out_map_entry.add_out_connector('OUT_inp')
    out_map_entry.add_out_connector('OUT_coeff')
    out_map_entry.add_out_connector('OUT_flx')
    out_map_entry.add_out_connector('OUT_fly')
    state.add_edge(out_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_coeff', tasklet, 'coeff_ij',
                   dace.Memlet.simple('coeff', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_flx', tasklet, 'flx_ij',
                   dace.Memlet.simple('flx', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_flx', tasklet, 'flx_im1j',
                   dace.Memlet.simple('flx', subset_str='i-1, j, k'))
    state.add_edge(out_map_entry, 'OUT_fly', tasklet, 'fly_ij',
                   dace.Memlet.simple('fly', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_fly', tasklet, 'fly_ijm1',
                   dace.Memlet.simple('fly', subset_str='i, j-1, k'))

    out_map_exit.add_in_connector('IN_out')
    state.add_edge(tasklet, 'out_ij', out_map_exit, 'IN_out',
                   dace.Memlet.simple('out', subset_str='i, j, k'))

    out_map_exit.add_out_connector('OUT_out')
    state.add_edge(
        out_map_exit, 'OUT_out', out_write, None,
        dace.Memlet.simple('out', subset_str='1:nx-1, 1:ny-1, 0:nz'))

    return sdfg


def generate_sdfg_flx_on_the_fly(name='hdiff_flx_on_the_fly'):
    sdfg = dace.SDFG(name)

    nx = dace.symbol('nx', dtype=dace.dtypes.int32)
    ny = dace.symbol('ny', dtype=dace.dtypes.int32)
    nz = dace.symbol('nz', dtype=dace.dtypes.int32)

    inp = sdfg.add_array('inp', (nx, ny, nz), dace.dtypes.float64)
    coeff = sdfg.add_array('coeff', (nx, ny, nz), dace.dtypes.float64)
    out = sdfg.add_array('out', (nx, ny, nz), dace.dtypes.float64)

    lap = sdfg.add_transient('lap', (nx, ny, nz), dace.dtypes.float64)
    fly = sdfg.add_transient('fly', (nx, ny, nz), dace.dtypes.float64)

    state = sdfg.add_state()

    inp_read = state.add_read('inp')
    coeff_read = state.add_read('coeff')
    out_write = state.add_write('out')

    # lap
    lap_map_entry, lap_map_exit = state.add_map('lap_map',
                                                ndrange=dict(i='1:nx-1',
                                                             j='1:ny-1',
                                                             k='0:nz'))

    tasklet = state.add_tasklet(
        'lap_tasklet',
        inputs={'inp_im1j', 'inp_ijm1', 'inp_ij', 'inp_ip1j', 'inp_ijp1'},
        outputs={'lap_ij'},
        code='lap_ij = 4 * inp_ij - (inp_im1j + inp_ijm1 + inp_ip1j + inp_ijp1)'
    )

    lap_map_entry.add_in_connector('IN_inp')
    state.add_edge(inp_read, None, lap_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', subset_str='0:nx, 0:ny, 0:nz'))

    lap_map_entry.add_out_connector('OUT_inp')
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_im1j',
                   dace.Memlet.simple('inp', subset_str='i-1, j, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ip1j',
                   dace.Memlet.simple('inp', subset_str='i+1, j, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ijm1',
                   dace.Memlet.simple('inp', subset_str='i, j-1, k'))
    state.add_edge(lap_map_entry, 'OUT_inp', tasklet, 'inp_ijp1',
                   dace.Memlet.simple('inp', subset_str='i, j+1, k'))

    lap_map_exit.add_in_connector('IN_lap')
    state.add_edge(tasklet, 'lap_ij', lap_map_exit, 'IN_lap',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))

    lap_map_exit.add_out_connector('OUT_lap')
    lap_access = state.add_access('lap')
    state.add_edge(
        lap_map_exit, 'OUT_lap', lap_access, None,
        dace.Memlet.simple('lap', subset_str='1:nx-1, 1:ny-1, 0:nz'))

    # fly
    fly_map_entry, fly_map_exit = state.add_map('fly_map',
                                                ndrange=dict(i='0:nx',
                                                             j='0:ny-1',
                                                             k='0:nz'))
    tasklet = state.add_tasklet(
        'fly_tasklet',
        inputs={'inp_ij', 'inp_ijp1', 'lap_ij', 'lap_ijp1'},
        outputs={'fly_ij'},
        code='fly_ij = lap_ijp1 - lap_ij\n'
        'if fly_ij * (inp_ijp1 - inp_ij) > 0: fly_ij = 0')

    fly_map_entry.add_in_connector('IN_inp')
    fly_map_entry.add_in_connector('IN_lap')
    state.add_edge(inp_read, None, fly_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', subset_str='1:nx, 0:ny, 0:nz'))
    state.add_edge(lap_access, None, fly_map_entry, 'IN_lap',
                   dace.Memlet.simple('lap', subset_str='1:nx, 0:ny, 0:nz'))

    fly_map_entry.add_out_connector('OUT_inp')
    fly_map_entry.add_out_connector('OUT_lap')
    state.add_edge(fly_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(fly_map_entry, 'OUT_inp', tasklet, 'inp_ijp1',
                   dace.Memlet.simple('inp', subset_str='i, j+1, k'))
    state.add_edge(fly_map_entry, 'OUT_lap', tasklet, 'lap_ij',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))
    state.add_edge(fly_map_entry, 'OUT_lap', tasklet, 'lap_ijp1',
                   dace.Memlet.simple('lap', subset_str='i, j+1, k'))

    fly_map_exit.add_in_connector('IN_fly')
    state.add_edge(tasklet, 'fly_ij', fly_map_exit, 'IN_fly',
                   dace.Memlet.simple('fly', subset_str='i, j, k'))

    fly_map_exit.add_out_connector('OUT_fly')
    fly_access = state.add_access('fly')
    state.add_edge(fly_map_exit, 'OUT_fly', fly_access, None,
                   dace.Memlet.simple('fly', subset_str='0:nx, 0:ny-1, 0:nz'))

    # out
    out_map_entry, out_map_exit = state.add_map('out_map',
                                                ndrange=dict(i='1:nx-1',
                                                             j='1:ny-1',
                                                             k='0:nz'))
    flx_ij_tasklet = state.add_tasklet(
        'flx_ij_tasklet',
        inputs={
            'flx_ij_tasklet_inp_ij', 'flx_ij_tasklet_inp_ip1j',
            'flx_ij_tasklet_lap_ij', 'flx_ij_tasklet_lap_ip1j'
        },
        outputs={'flx_ij_tasklet_flx_ij'},
        code='flx_ij_tasklet_flx_ij = '
        'flx_ij_tasklet_lap_ip1j - flx_ij_tasklet_lap_ij\n'
        'if flx_ij_tasklet_flx_ij * '
        '(flx_ij_tasklet_inp_ip1j - flx_ij_tasklet_inp_ij) > 0:\n'
        '    flx_ij_tasklet_flx_ij = 0')
    flx_im1j_tasklet = state.add_tasklet(
        'flx_im1j_tasklet',
        inputs={
            'flx_im1j_tasklet_inp_ij', 'flx_im1j_tasklet_inp_ip1j',
            'flx_im1j_tasklet_lap_ij', 'flx_im1j_tasklet_lap_ip1j'
        },
        outputs={'flx_im1j_tasklet_flx_ij'},
        code='flx_im1j_tasklet_flx_ij = '
        'flx_im1j_tasklet_lap_ip1j - flx_im1j_tasklet_lap_ij\n'
        'if flx_im1j_tasklet_flx_ij * '
        '(flx_im1j_tasklet_inp_ip1j - flx_im1j_tasklet_inp_ij) > 0:\n'
        '    flx_im1j_tasklet_flx_ij = 0')
    tasklet = state.add_tasklet(
        'out_tasklet',
        inputs={
            'out_tasklet_inp_ij', 'out_tasklet_coeff_ij',
            'out_tasklet_flx_im1j', 'out_tasklet_flx_ij',
            'out_tasklet_fly_ijm1', 'out_tasklet_fly_ij'
        },
        outputs={'out_tasklet_out_ij'},
        code='out_tasklet_out_ij = out_tasklet_inp_ij - out_tasklet_coeff_ij * '
        '(out_tasklet_flx_ij - out_tasklet_flx_im1j + '
        'out_tasklet_fly_ij - out_tasklet_fly_ijm1)')

    out_map_entry.add_in_connector('IN_inp')
    out_map_entry.add_in_connector('IN_coeff')
    out_map_entry.add_in_connector('IN_lap')
    out_map_entry.add_in_connector('IN_fly')
    state.add_edge(inp_read, None, out_map_entry, 'IN_inp',
                   dace.Memlet.simple('inp', '0:nx, 1:ny, 0:nz'))
    state.add_edge(coeff_read, None, out_map_entry, 'IN_coeff',
                   dace.Memlet.simple('coeff', '1:nx, 1:ny, 0:nz'))
    state.add_edge(lap_access, None, out_map_entry, 'IN_lap',
                   dace.Memlet.simple('lap', '0:nx, 0:ny, 0:nz'))
    state.add_edge(fly_access, None, out_map_entry, 'IN_fly',
                   dace.Memlet.simple('fly', '0:nx, 1:ny, 0:nz'))

    out_map_entry.add_out_connector('OUT_inp')
    out_map_entry.add_out_connector('OUT_coeff')
    out_map_entry.add_out_connector('OUT_lap')
    out_map_entry.add_out_connector('OUT_fly')
    state.add_edge(out_map_entry, 'OUT_inp', tasklet, 'out_tasklet_inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_coeff', tasklet, 'out_tasklet_coeff_ij',
                   dace.Memlet.simple('coeff', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_fly', tasklet, 'out_tasklet_fly_ij',
                   dace.Memlet.simple('fly', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_fly', tasklet, 'out_tasklet_fly_ijm1',
                   dace.Memlet.simple('fly', subset_str='i, j-1, k'))

    state.add_edge(out_map_entry, 'OUT_inp', flx_ij_tasklet,
                   'flx_ij_tasklet_inp_ij',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_inp', flx_ij_tasklet,
                   'flx_ij_tasklet_inp_ip1j',
                   dace.Memlet.simple('inp', subset_str='i+1, j, k'))
    state.add_edge(out_map_entry, 'OUT_lap', flx_ij_tasklet,
                   'flx_ij_tasklet_lap_ij',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_lap', flx_ij_tasklet,
                   'flx_ij_tasklet_lap_ip1j',
                   dace.Memlet.simple('lap', subset_str='i+1, j, k'))

    state.add_edge(out_map_entry, 'OUT_inp', flx_im1j_tasklet,
                   'flx_im1j_tasklet_inp_ij',
                   dace.Memlet.simple('inp', subset_str='i-1, j, k'))
    state.add_edge(out_map_entry, 'OUT_inp', flx_im1j_tasklet,
                   'flx_im1j_tasklet_inp_ip1j',
                   dace.Memlet.simple('inp', subset_str='i, j, k'))
    state.add_edge(out_map_entry, 'OUT_lap', flx_im1j_tasklet,
                   'flx_im1j_tasklet_lap_ij',
                   dace.Memlet.simple('lap', subset_str='i-1, j, k'))
    state.add_edge(out_map_entry, 'OUT_lap', flx_im1j_tasklet,
                   'flx_im1j_tasklet_lap_ip1j',
                   dace.Memlet.simple('lap', subset_str='i, j, k'))

    sdfg.add_scalar('flx_ij', dace.dtypes.float64, transient=True)
    sdfg.add_scalar('flx_im1j', dace.dtypes.float64, transient=True)

    flx_ij_access = state.add_access('flx_ij')
    flx_im1j_access = state.add_access('flx_im1j')

    state.add_edge(flx_ij_tasklet, 'flx_ij_tasklet_flx_ij', flx_ij_access,
                   None, dace.Memlet('flx_ij'))
    state.add_edge(flx_im1j_tasklet, 'flx_im1j_tasklet_flx_ij',
                   flx_im1j_access, None, dace.Memlet('flx_im1j'))

    state.add_edge(flx_ij_access, None, tasklet, 'out_tasklet_flx_ij',
                   dace.Memlet('flx_ij'))
    state.add_edge(flx_im1j_access, None, tasklet, 'out_tasklet_flx_im1j',
                   dace.Memlet('flx_im1j'))

    out_map_exit.add_in_connector('IN_out')
    state.add_edge(tasklet, 'out_tasklet_out_ij', out_map_exit, 'IN_out',
                   dace.Memlet.simple('out', subset_str='i, j, k'))

    out_map_exit.add_out_connector('OUT_out')
    state.add_edge(
        out_map_exit, 'OUT_out', out_write, None,
        dace.Memlet.simple('out', subset_str='1:nx-1, 1:ny-1, 0:nz'))

    return sdfg


def test_horizontal_diffusion_raw(horizontal_diffusion):
    ref_inp, ref_coeff, ref_out = horizontal_diffusion

    sdfg = generate_sdfg()

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             coeff=ref_coeff,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[2:-2, 2:-2, :], ref_out[2:-2, 2:-2, :])


def test_horizontal_diffusion_flx_on_the_fly(horizontal_diffusion):
    ref_inp, ref_coeff, ref_out = horizontal_diffusion

    sdfg = generate_sdfg_flx_on_the_fly()

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             coeff=ref_coeff,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[2:-2, 2:-2, :], ref_out[2:-2, 2:-2, :])


def test_horizontal_diffusion_transformed_flx_otf(horizontal_diffusion):
    ref_inp, ref_coeff, ref_out = horizontal_diffusion

    sdfg = generate_sdfg('hdiff_transformed_flx_otf')

    from daceperiments.transforms import OnTheFlyMapFusion
    sdfg.apply_transformations(OnTheFlyMapFusion, validate=True)

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             coeff=ref_coeff,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[2:-2, 2:-2, :], ref_out[2:-2, 2:-2, :])


def test_horizontal_diffusion_transformed_flx_fly_otf(horizontal_diffusion):
    ref_inp, ref_coeff, ref_out = horizontal_diffusion

    sdfg = generate_sdfg('hdiff_transformed_flx_fly_otf')

    from daceperiments.transforms import OnTheFlyMapFusion
    sdfg.apply_transformations(OnTheFlyMapFusion, validate=True)
    sdfg.apply_transformations(OnTheFlyMapFusion, validate=True)

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             coeff=ref_coeff,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[2:-2, 2:-2, :], ref_out[2:-2, 2:-2, :])


def test_horizontal_diffusion_transformed_all_otf(horizontal_diffusion):
    ref_inp, ref_coeff, ref_out = horizontal_diffusion

    sdfg = generate_sdfg('hdiff_transformed_all_otf')

    from daceperiments.transforms import OnTheFlyMapFusion
    sdfg.apply_transformations_repeated(OnTheFlyMapFusion, validate=True)

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             coeff=ref_coeff,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[2:-2, 2:-2, :], ref_out[2:-2, 2:-2, :])
