import dace
import daceperiments.transforms
import numpy as np
import pytest


@pytest.fixture()
def inclusive_scan_backward():
    n = 10
    inp = np.random.uniform(size=n)
    out = np.zeros_like(inp)
    buf = np.zeros_like(inp)

    out[-1] = buf[-1] = inp[-1]
    for i in range(n - 2, -1, -1):
        buf[i] = buf[i + 1] + inp[i]
        out[i] = buf[i]

    return inp, out


def generate_sdfg(name):
    sdfg = dace.SDFG(name)
    n = dace.symbol('n', dtype=dace.dtypes.int32)

    inp = sdfg.add_array('inp', (n, ), dace.dtypes.float64)
    out = sdfg.add_array('out', (n, ), dace.dtypes.float64)
    buf = sdfg.add_transient('buf', (n, ), dace.dtypes.float64)

    # initial iteration
    before_state = sdfg.add_state(is_start_state=True)

    inp_read = before_state.add_read('inp')
    out_write = before_state.add_write('out')
    buf_write = before_state.add_write('buf')

    tasklet = before_state.add_tasklet(
        'before_tasklet',
        inputs={'inp_read'},
        outputs={'out_write', 'buf_write'},
        code='buf_write = inp_read; out_write = inp_read')

    before_state.add_edge(inp_read, None, tasklet, 'inp_read',
                          dace.Memlet.simple('inp', subset_str='n - 1'))
    before_state.add_edge(tasklet, 'out_write', out_write, None,
                          dace.Memlet.simple('out', subset_str='n - 1'))
    before_state.add_edge(tasklet, 'buf_write', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='n - 1'))

    # other iterations
    loop_state = sdfg.add_state()

    inp_read = loop_state.add_read('inp')
    buf_read = loop_state.add_read('buf')
    out_write = loop_state.add_write('out')
    buf_write = loop_state.add_write('buf')

    tasklet = loop_state.add_tasklet(
        'loop_tasklet',
        inputs={'inp_read', 'buf_read'},
        outputs={'out_write', 'buf_write'},
        code='buf_write = buf_read + inp_read; out_write = buf_write')

    loop_state.add_edge(inp_read, None, tasklet, 'inp_read',
                        dace.Memlet.simple('inp', subset_str='i'))
    loop_state.add_edge(buf_read, None, tasklet, 'buf_read',
                        dace.Memlet.simple('buf', subset_str='i + 1'))
    loop_state.add_edge(tasklet, 'out_write', out_write, None,
                        dace.Memlet.simple('out', subset_str='i'))
    loop_state.add_edge(tasklet, 'buf_write', buf_write, None,
                        dace.Memlet.simple('buf', subset_str='i'))

    # loop
    sdfg.add_loop(before_state=before_state,
                  loop_state=loop_state,
                  after_state=None,
                  initialize_expr='n - 2',
                  increment_expr='i - 1',
                  condition_expr='i >= 0',
                  loop_var='i')

    sdfg.validate()
    return sdfg


def test_raw_dace(inclusive_scan_backward):
    ref_inp, ref_out = inclusive_scan_backward

    sdfg = generate_sdfg('raw_inclusive_scan_backward')
    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp, out=out, n=out.size)

    np.testing.assert_allclose(out, ref_out)


def test_transform(inclusive_scan_backward):
    ref_inp, ref_out = inclusive_scan_backward

    sdfg = generate_sdfg('transformed_inclusive_scan_backward')

    assert sdfg.apply_transformations(
        daceperiments.transforms.BasicRegisterCache,
        dict(array='buf'),
        validate=True) == 1

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    compiled(inp=ref_inp, out=out, n=out.size)

    np.testing.assert_allclose(out, ref_out)
