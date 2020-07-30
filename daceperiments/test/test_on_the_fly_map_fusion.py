import dace
import numpy as np
import pytest


@pytest.fixture()
def basic_diff():
    nx, ny, nz = 10, 13, 17

    inp = np.random.uniform(size=(nx, ny, nz))
    tmp1 = 2 * inp
    tmp2 = 2 * tmp1 + inp
    out = np.zeros_like(inp)
    out[1:-1, :, :] = tmp2[2:, :, :] - tmp2[:-2, :, :]

    return inp, out


def generate_sdfg(name='basic_diff'):
    sdfg = dace.SDFG(name)

    nx = dace.symbol('nx', dtype=dace.dtypes.int32)
    ny = dace.symbol('ny', dtype=dace.dtypes.int32)
    nz = dace.symbol('nz', dtype=dace.dtypes.int32)

    inp = sdfg.add_array('inp', (nx, ny, nz), dace.dtypes.float64)
    out = sdfg.add_array('out', (nx, ny, nz), dace.dtypes.float64)

    tmp1 = sdfg.add_transient('tmp1', (nx, ny, nz), dace.dtypes.float64)
    tmp2 = sdfg.add_transient('tmp2', (nx, ny, nz), dace.dtypes.float64)

    state = sdfg.add_state()

    inp_read = state.add_read('inp')
    out_write = state.add_write('out')
    tmp1_access = state.add_access('tmp1')
    tmp2_access = state.add_access('tmp2')

    tmp1_map_entry, tmp1_map_exit = state.add_map('tmp1_map',
                                                  ndrange=dict(i='0:nx',
                                                               j='0:ny',
                                                               k='0:nz'))

    tasklet = state.add_tasklet('tmp1_tasklet',
                                inputs={'inp_ij'},
                                outputs={'tmp1_ij'},
                                code='tmp1_ij = 2 * inp_ij')

    tmp1_map_entry.add_in_connector('IN_inp')
    state.add_edge(inp_read, None, tmp1_map_entry, 'IN_inp',
                   dace.Memlet('inp[0:nx, 0:ny, 0:nz]'))
    tmp1_map_entry.add_out_connector('OUT_inp')
    state.add_edge(tmp1_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet('inp[i, j, k]'))
    tmp1_map_exit.add_in_connector('IN_tmp1')
    state.add_edge(tasklet, 'tmp1_ij', tmp1_map_exit, 'IN_tmp1',
                   dace.Memlet('tmp1[i, j, k]'))
    tmp1_map_exit.add_out_connector('OUT_tmp1')
    state.add_edge(tmp1_map_exit, 'OUT_tmp1', tmp1_access, None,
                   dace.Memlet('tmp1[0:nx, 0:ny, 0:nz]'))

    tmp2_map_entry, tmp2_map_exit = state.add_map('tmp2_map',
                                                  ndrange=dict(i='0:nx',
                                                               j='0:ny',
                                                               k='0:nz'))

    tasklet = state.add_tasklet('tmp2_tasklet',
                                inputs={'tmp1_ij', 'inp_ij'},
                                outputs={'tmp2_ij'},
                                code='tmp2_ij = 2 * tmp1_ij + inp_ij')

    tmp2_map_entry.add_in_connector('IN_inp')
    state.add_edge(inp_read, None, tmp2_map_entry, 'IN_inp',
                   dace.Memlet('inp[0:nx, 0:ny, 0:nz]'))
    tmp2_map_entry.add_in_connector('IN_tmp1')
    state.add_edge(tmp1_access, None, tmp2_map_entry, 'IN_tmp1',
                   dace.Memlet('tmp1[0:nx, 0:ny, 0:nz]'))
    tmp2_map_entry.add_out_connector('OUT_inp')
    state.add_edge(tmp2_map_entry, 'OUT_inp', tasklet, 'inp_ij',
                   dace.Memlet('inp[i, j, k]'))
    tmp2_map_entry.add_out_connector('OUT_tmp1')
    state.add_edge(tmp2_map_entry, 'OUT_tmp1', tasklet, 'tmp1_ij',
                   dace.Memlet('tmp1[i, j, k]'))
    tmp2_map_exit.add_in_connector('IN_tmp2')
    state.add_edge(tasklet, 'tmp2_ij', tmp2_map_exit, 'IN_tmp2',
                   dace.Memlet('tmp2[i, j, k]'))
    tmp2_map_exit.add_out_connector('OUT_tmp2')
    state.add_edge(tmp2_map_exit, 'OUT_tmp2', tmp2_access, None,
                   dace.Memlet('tmp2[0:nx, 0:ny, 0:nz]'))

    out_map_entry, out_map_exit = state.add_map('out_map',
                                                ndrange=dict(i='1:nx-1',
                                                             j='0:ny',
                                                             k='0:nz'))

    tasklet = state.add_tasklet('out_tasklet',
                                inputs={'tmp2_im1j', 'tmp2_ip1j'},
                                outputs={'out_ij'},
                                code='out_ij = tmp2_ip1j - tmp2_im1j')
    out_map_entry.add_in_connector('IN_tmp2')
    state.add_edge(tmp2_access, None, out_map_entry, 'IN_tmp2',
                   dace.Memlet('tmp2[0:nx, 0:ny, 0:nz]'))
    out_map_entry.add_out_connector('OUT_tmp2')
    state.add_edge(out_map_entry, 'OUT_tmp2', tasklet, 'tmp2_im1j',
                   dace.Memlet('tmp2[i-1, j, k]'))
    state.add_edge(out_map_entry, 'OUT_tmp2', tasklet, 'tmp2_ip1j',
                   dace.Memlet('tmp2[i+1, j, k]'))
    out_map_exit.add_in_connector('IN_out')
    state.add_edge(tasklet, 'out_ij', out_map_exit, 'IN_out',
                   dace.Memlet('out[i, j, k]'))
    out_map_exit.add_out_connector('OUT_out')
    state.add_edge(out_map_exit, 'OUT_out', out_write, None,
                   dace.Memlet('out[0:nx, 0:ny, 0:nz]'))

    return sdfg


def test_basic_diff(basic_diff):
    ref_inp, ref_out = basic_diff

    sdfg = generate_sdfg()

    compiled = sdfg.compile(optimizer=False)
    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[1:-1, :, :], ref_out[1:-1, :, :])


def test_basic_diff_otf(basic_diff):
    ref_inp, ref_out = basic_diff

    sdfg = generate_sdfg('basic_diff_otf')

    from daceperiments.transforms import OnTheFlyMapFusion
    assert sdfg.apply_transformations_repeated(OnTheFlyMapFusion,
                                               validate=True) == 2

    compiled = sdfg.compile(optimizer=False)
    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out[1:-1, :, :], ref_out[1:-1, :, :])
