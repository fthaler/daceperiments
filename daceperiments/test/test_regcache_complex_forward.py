import dace
import daceperiments.transforms
import numpy as np
import pytest


@pytest.fixture()
def complex_scan_forward():
    n = 10
    inp = np.random.uniform(size=n)
    out = np.zeros_like(inp)
    buf = np.zeros_like(inp)

    buf[0] = inp[0]
    buf[1] = inp[0] / 2
    out[0] = buf[0]

    buf[2] = inp[1] / 2
    out[1] = buf[1] + buf[0] / 4

    for i in range(2, n - 1):
        buf[i + 1] = inp[i] / 2
        out[i] = buf[i] + buf[i - 1] / 4 + buf[i - 2] / 8

    out[n - 1] = buf[n - 1] + buf[n - 2] / 4 + buf[n - 3] / 8

    return inp, out


def generate_sdfg(name):
    sdfg = dace.SDFG(name)
    n = dace.symbol('n', dtype=dace.dtypes.int32)

    inp = sdfg.add_array('inp', (n, ), dace.dtypes.float64)
    out = sdfg.add_array('out', (n, ), dace.dtypes.float64)
    buf = sdfg.add_array('buf', (n, ), dace.dtypes.float64)

    # initial iteration
    before_state = sdfg.add_state(is_start_state=True)

    inp_read = before_state.add_read('inp')
    out_write = before_state.add_write('out')
    buf_write = before_state.add_write('buf')

    tasklet = before_state.add_tasklet(
        'before_tasklet',
        inputs={'inp_read'},
        outputs={'out_write', 'buf_write_0', 'buf_write_1'},
        code=
        'buf_write_0 = inp_read; buf_write_1 = inp_read / 2; out_write = inp_read'
    )

    before_state.add_edge(inp_read, None, tasklet, 'inp_read',
                          dace.Memlet.simple('inp', subset_str='0'))
    before_state.add_edge(tasklet, 'out_write', out_write, None,
                          dace.Memlet.simple('out', subset_str='0'))
    before_state.add_edge(tasklet, 'buf_write_0', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='0'))
    before_state.add_edge(tasklet, 'buf_write_1', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='1'))

    # second iteration
    before_state = sdfg.add_state_after(before_state)

    inp_read = before_state.add_read('inp')
    buf_read = before_state.add_read('buf')
    out_write = before_state.add_write('out')
    buf_write = before_state.add_write('buf')

    tasklet = before_state.add_tasklet(
        'before_tasklet',
        inputs={'inp_read', 'buf_read_0', 'buf_read_1'},
        outputs={'out_write', 'buf_write'},
        code='buf_write = inp_read / 2; out_write = buf_read_1 + buf_read_0 / 4'
    )

    before_state.add_edge(inp_read, None, tasklet, 'inp_read',
                          dace.Memlet.simple('inp', subset_str='1'))
    before_state.add_edge(buf_read, None, tasklet, 'buf_read_0',
                          dace.Memlet.simple('buf', subset_str='0'))
    before_state.add_edge(buf_read, None, tasklet, 'buf_read_1',
                          dace.Memlet.simple('buf', subset_str='1'))
    before_state.add_edge(tasklet, 'out_write', out_write, None,
                          dace.Memlet.simple('out', subset_str='1'))
    before_state.add_edge(tasklet, 'buf_write', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='2'))

    # other iterations
    loop_state = sdfg.add_state()

    inp_read = loop_state.add_read('inp')
    buf_read = loop_state.add_read('buf')
    out_write = loop_state.add_write('out')
    buf_write = loop_state.add_write('buf')

    tasklet = loop_state.add_tasklet(
        'loop_tasklet',
        inputs={'inp_read', 'buf_read_0', 'buf_read_1', 'buf_read_2'},
        outputs={'out_write', 'buf_write'},
        code='buf_write = inp_read / 2; '
        'out_write = buf_read_2 + buf_read_1 / 4 + buf_read_0 / 8')

    loop_state.add_edge(inp_read, None, tasklet, 'inp_read',
                        dace.Memlet.simple('inp', subset_str='i'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read_0',
                        dace.Memlet.simple('buf', subset_str='i - 2'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read_1',
                        dace.Memlet.simple('buf', subset_str='i - 1'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read_2',
                        dace.Memlet.simple('buf', subset_str='i'))
    loop_state.add_edge(tasklet, 'out_write', out_write, None,
                        dace.Memlet.simple('out', subset_str='i'))
    loop_state.add_edge(tasklet, 'buf_write', buf_write, None,
                        dace.Memlet.simple('buf', subset_str='i + 1'))

    # last iteration
    after_state = sdfg.add_state()

    buf_read = after_state.add_read('buf')
    out_write = after_state.add_write('out')

    tasklet = after_state.add_tasklet(
        'loop_tasklet',
        inputs={'buf_read_0', 'buf_read_1', 'buf_read_2'},
        outputs={'out_write'},
        code='out_write = buf_read_2 + buf_read_1 / 4 + buf_read_0 / 8')

    after_state.add_edge(buf_read, None, tasklet, 'buf_read_0',
                         dace.Memlet.simple('buf', subset_str=f'{n} - 3'))
    after_state.add_edge(buf_read, None, tasklet, 'buf_read_1',
                         dace.Memlet.simple('buf', subset_str=f'{n} - 2'))
    after_state.add_edge(buf_read, None, tasklet, 'buf_read_2',
                         dace.Memlet.simple('buf', subset_str=f'{n} - 1'))
    after_state.add_edge(tasklet, 'out_write', out_write, None,
                         dace.Memlet.simple('out', subset_str=f'{n} - 1'))

    # loop
    sdfg.add_loop(before_state=before_state,
                  loop_state=loop_state,
                  after_state=after_state,
                  initialize_expr='2',
                  increment_expr='i + 1',
                  condition_expr=f'i < {n - 1}',
                  loop_var='i')

    sdfg.validate()
    return sdfg


def test_raw_dace(complex_scan_forward):
    ref_inp, ref_out = complex_scan_forward

    sdfg = generate_sdfg('raw_complex_scan_forward')
    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    buf = np.zeros_like(out)
    compiled(inp=ref_inp, out=out, buf=buf, n=out.size)

    np.testing.assert_allclose(out, ref_out)


def test_transform(complex_scan_forward):
    ref_inp, ref_out = complex_scan_forward

    sdfg = generate_sdfg('transformed_complex_scan_forward')

    assert sdfg.apply_transformations(
        daceperiments.transforms.BasicRegisterCache,
        dict(array='buf'),
        validate=True) == 1

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    buf = np.zeros_like(out, shape=(4, ))
    compiled(inp=ref_inp, out=out, buf=buf, n=out.size)

    np.testing.assert_allclose(out, ref_out)
