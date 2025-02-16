import dace
import daceperiments.transforms
import numpy as np
import pytest


@pytest.fixture()
def complex_scan_forward_nd():
    nx, ny, nz = 10, 13, 17
    inp = np.random.uniform(size=(nx, ny, nz))
    out = np.zeros_like(inp)
    buf = np.zeros_like(inp)

    buf[:, 0, :] = inp[:, 0, :]
    buf[:, 1, :] = inp[:, 0, :] / 2
    out[:, 0, :] = buf[:, 0, :]

    buf[:, 2, :] = inp[:, 1, :] / 2
    out[:, 1, :] = buf[:, 1, :] + buf[:, 0, :] / 4

    for j in range(2, ny - 1):
        buf[:, j + 1, :] = inp[:, j, :] / 2
        out[:, j, :] = (buf[:, j, :] + buf[:, j - 1, :] / 4 +
                        buf[:, j - 2, :] / 8)

    out[:, ny - 1, :] = (buf[:, ny - 1, :] + buf[:, ny - 2, :] / 4 +
                         buf[:, ny - 3, :] / 8)

    return inp, out


def generate_sdfg(name):
    sdfg = dace.SDFG(name)
    nested_sdfg = dace.SDFG(f'nested_{name}')

    nx = dace.symbol('nx', dtype=dace.dtypes.int32)
    ny = dace.symbol('ny', dtype=dace.dtypes.int32)
    nz = dace.symbol('nz', dtype=dace.dtypes.int32)

    nested_sdfg.add_symbol('nx', nx.dtype)
    nested_sdfg.add_symbol('ny', ny.dtype)
    nested_sdfg.add_symbol('nz', nz.dtype)

    inp = sdfg.add_array('inp', (nx, ny, nz), dace.dtypes.float64)
    out = sdfg.add_array('out', (nx, ny, nz), dace.dtypes.float64)

    state = sdfg.add_state(is_start_state=True)

    inp_read = state.add_read('inp')
    out_write = state.add_write('out')

    nested_tasklet = state.add_nested_sdfg(nested_sdfg,
                                           sdfg,
                                           inputs={'inp_column'},
                                           outputs={'out_column'})
    map_entry, map_exit = state.add_map('map',
                                        ndrange=dict(i='0:nx', k='0:nz'))
    map_entry.map.collapse = 2

    state.add_memlet_path(inp_read,
                          map_entry,
                          nested_tasklet,
                          dst_conn='inp_column',
                          memlet=dace.Memlet.simple('inp',
                                                    subset_str='i, 0:ny, k'))
    state.add_memlet_path(nested_tasklet,
                          map_exit,
                          out_write,
                          src_conn='out_column',
                          memlet=dace.Memlet.simple('out',
                                                    subset_str='i, 0:ny, k'))

    inp = nested_sdfg.add_array('inp_column', (1, ny, 1),
                                dace.dtypes.float64,
                                strides=inp[1].strides,
                                total_size=inp[1].total_size)
    out = nested_sdfg.add_array('out_column', (1, ny, 1),
                                dace.dtypes.float64,
                                strides=out[1].strides,
                                total_size=out[1].total_size)
    buf = nested_sdfg.add_transient('buf', (1, ny, 1), dace.dtypes.float64)

    # initial iteration
    before_state = nested_sdfg.add_state(is_start_state=True)

    inp_read = before_state.add_read('inp_column')
    out_write = before_state.add_write('out_column')
    buf_write = before_state.add_write('buf')

    tasklet = before_state.add_tasklet(
        'before_tasklet',
        inputs={'inp_read'},
        outputs={'out_write', 'buf_write_0', 'buf_write_1'},
        code='buf_write_0 = inp_read; '
        'buf_write_1 = inp_read / 2; '
        'out_write = inp_read')

    before_state.add_edge(
        inp_read, None, tasklet, 'inp_read',
        dace.Memlet.simple('inp_column', subset_str='0, 0, 0'))
    before_state.add_edge(
        tasklet, 'out_write', out_write, None,
        dace.Memlet.simple('out_column', subset_str='0, 0, 0'))
    before_state.add_edge(tasklet, 'buf_write_0', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='0, 0, 0'))
    before_state.add_edge(tasklet, 'buf_write_1', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='0, 1, 0'))

    # second iteration
    before_state = nested_sdfg.add_state_after(before_state)

    inp_read = before_state.add_read('inp_column')
    buf_read = before_state.add_read('buf')
    out_write = before_state.add_write('out_column')
    buf_write = before_state.add_write('buf')

    tasklet = before_state.add_tasklet(
        'before_tasklet',
        inputs={'inp_read', 'buf_read_0', 'buf_read_1'},
        outputs={'out_write', 'buf_write'},
        code='buf_write = inp_read / 2; out_write = buf_read_1 + buf_read_0 / 4'
    )

    before_state.add_edge(
        inp_read, None, tasklet, 'inp_read',
        dace.Memlet.simple('inp_column', subset_str='0, 1, 0'))
    before_state.add_edge(buf_read, None, tasklet, 'buf_read_0',
                          dace.Memlet.simple('buf', subset_str='0, 0, 0'))
    before_state.add_edge(buf_read, None, tasklet, 'buf_read_1',
                          dace.Memlet.simple('buf', subset_str='0, 1, 0'))
    before_state.add_edge(
        tasklet, 'out_write', out_write, None,
        dace.Memlet.simple('out_column', subset_str='0, 1, 0'))
    before_state.add_edge(tasklet, 'buf_write', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='0, 2, 0'))

    # other iterations
    loop_state = nested_sdfg.add_state()

    inp_read = loop_state.add_read('inp_column')
    buf_read = loop_state.add_read('buf')
    out_write = loop_state.add_write('out_column')
    buf_write = loop_state.add_write('buf')

    tasklet = loop_state.add_tasklet(
        'loop_tasklet',
        inputs={'inp_read', 'buf_read_0', 'buf_read_1', 'buf_read_2'},
        outputs={'out_write', 'buf_write'},
        code='buf_write = inp_read / 2; '
        'out_write = buf_read_2 + buf_read_1 / 4 + buf_read_0 / 8')

    loop_state.add_edge(inp_read, None, tasklet, 'inp_read',
                        dace.Memlet.simple('inp_column', subset_str='0, j, 0'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read_0',
                        dace.Memlet.simple('buf', subset_str='0, j - 2, 0'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read_1',
                        dace.Memlet.simple('buf', subset_str='0, j - 1, 0'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read_2',
                        dace.Memlet.simple('buf', subset_str='0, j, 0'))
    loop_state.add_edge(tasklet, 'out_write', out_write, None,
                        dace.Memlet.simple('out_column', subset_str='0, j, 0'))
    loop_state.add_edge(tasklet, 'buf_write', buf_write, None,
                        dace.Memlet.simple('buf', subset_str='0, j + 1, 0'))

    # last iteration
    after_state = nested_sdfg.add_state()

    buf_read = after_state.add_read('buf')
    out_write = after_state.add_write('out_column')

    tasklet = after_state.add_tasklet(
        'loop_tasklet',
        inputs={'buf_read_0', 'buf_read_1', 'buf_read_2'},
        outputs={'out_write'},
        code='out_write = buf_read_2 + buf_read_1 / 4 + buf_read_0 / 8')

    after_state.add_edge(buf_read, None, tasklet, 'buf_read_0',
                         dace.Memlet.simple('buf', subset_str='0, ny - 3, 0'))
    after_state.add_edge(buf_read, None, tasklet, 'buf_read_1',
                         dace.Memlet.simple('buf', subset_str='0, ny - 2, 0'))
    after_state.add_edge(buf_read, None, tasklet, 'buf_read_2',
                         dace.Memlet.simple('buf', subset_str='0, ny - 1, 0'))
    after_state.add_edge(
        tasklet, 'out_write', out_write, None,
        dace.Memlet.simple('out_column', subset_str='0, ny - 1, 0'))

    # loop
    nested_sdfg.add_loop(before_state=before_state,
                         loop_state=loop_state,
                         after_state=after_state,
                         initialize_expr='2',
                         increment_expr='j + 1',
                         condition_expr='j < ny - 1',
                         loop_var='j')

    sdfg.validate()
    return sdfg


def test_raw_dace(complex_scan_forward_nd):
    ref_inp, ref_out = complex_scan_forward_nd

    sdfg = generate_sdfg('raw_complex_scan_forward_nd')
    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out, ref_out)


def test_transform(complex_scan_forward_nd):
    ref_inp, ref_out = complex_scan_forward_nd

    sdfg = generate_sdfg('transformed_complex_scan_forward_nd')

    assert sdfg.apply_transformations(
        daceperiments.transforms.BasicRegisterCache,
        dict(array='buf'),
        validate=True) == 1

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp,
             out=out,
             nx=out.shape[0],
             ny=out.shape[1],
             nz=out.shape[2])

    np.testing.assert_allclose(out, ref_out)
