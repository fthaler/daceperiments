import dace
import daceperiments.transforms
import numpy as np
import pytest


@pytest.fixture()
def inclusive_scan():
    n = 10
    inp = np.random.uniform(size=n)
    out = np.zeros_like(inp)
    buf = np.zeros_like(inp)

    out[0] = buf[0] = inp[0]
    for i in range(1, n):
        buf[i] = buf[i - 1] + inp[i]
        out[i] = buf[i]

    return inp, out


def generate_sdfg(name, n):
    sdfg = dace.SDFG(name)

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
        outputs={'out_write', 'buf_write'},
        code='buf_write = inp_read; out_write = inp_read')

    before_state.add_edge(inp_read, None, tasklet, 'inp_read',
                          dace.Memlet.simple('inp', subset_str='i'))
    before_state.add_edge(tasklet, 'out_write', out_write, None,
                          dace.Memlet.simple('out', subset_str='i'))
    before_state.add_edge(tasklet, 'buf_write', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='i'))

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
                        dace.Memlet.simple('buf', subset_str='i - 1'))
    loop_state.add_edge(tasklet, 'out_write', out_write, None,
                        dace.Memlet.simple('out', subset_str='i'))
    loop_state.add_edge(tasklet, 'buf_write', buf_write, None,
                        dace.Memlet.simple('buf', subset_str='i'))

    # loop
    sdfg.add_loop(before_state=before_state,
                  loop_state=loop_state,
                  after_state=None,
                  initialize_expr='1',
                  increment_expr='i + 1',
                  condition_expr=f'i < {n}',
                  loop_var='i')

    sdfg.validate()
    return sdfg


def test_raw_dace(inclusive_scan):
    ref_inp, ref_out = inclusive_scan

    sdfg = generate_sdfg('raw_inclusive_scan', ref_inp.size)
    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    buf = np.zeros_like(out)
    compiled(inp=ref_inp, out=out, buf=buf)

    np.testing.assert_allclose(out, ref_out)


def test_expected(inclusive_scan):
    ref_inp, ref_out = inclusive_scan
    n = ref_inp.size

    sdfg = dace.SDFG('expected')

    inp = sdfg.add_array('inp', (n, ), dace.dtypes.float64)
    out = sdfg.add_array('out', (n, ), dace.dtypes.float64)
    buf = sdfg.add_array('buf', (2, ),
                         dace.dtypes.float64,
                         storage=dace.StorageType.Register)

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
                          dace.Memlet.simple('inp', subset_str='i'))
    before_state.add_edge(tasklet, 'out_write', out_write, None,
                          dace.Memlet.simple('out', subset_str='i'))
    before_state.add_edge(tasklet, 'buf_write', buf_write, None,
                          dace.Memlet.simple('buf', subset_str='1'))

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
                        dace.Memlet.simple('buf', subset_str='0'))
    loop_state.add_edge(tasklet, 'out_write', out_write, None,
                        dace.Memlet.simple('out', subset_str='i'))
    loop_state.add_edge(tasklet, 'buf_write', buf_write, None,
                        dace.Memlet.simple('buf', subset_str='1'))

    shift_before_state = sdfg.add_state_after(before_state)
    buf_read = shift_before_state.add_read('buf')
    buf_write = shift_before_state.add_write('buf')

    shift_before_state.add_edge(
        buf_read, None, buf_write, None,
        dace.Memlet.simple('buf', subset_str='1', other_subset_str='0'))

    shift_loop_state = sdfg.add_state_after(loop_state)
    buf_read = shift_loop_state.add_read('buf')
    buf_write = shift_loop_state.add_write('buf')

    shift_loop_state.add_edge(
        buf_read, None, buf_write, None,
        dace.Memlet.simple('buf', subset_str='1', other_subset_str='0'))

    # loop
    sdfg.add_loop(before_state=shift_before_state,
                  loop_state=loop_state,
                  loop_end_state=shift_loop_state,
                  after_state=None,
                  initialize_expr='1',
                  increment_expr='i + 1',
                  condition_expr=f'i < {n}',
                  loop_var='i')

    sdfg.validate()
    sdfg.save('expected.sdfg')

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    buf = np.zeros_like(out)
    compiled(inp=ref_inp, out=out, buf=buf)

    np.testing.assert_allclose(out, ref_out)


def test_transform(inclusive_scan):
    ref_inp, ref_out = inclusive_scan

    sdfg = generate_sdfg('transformed_inclusive_scan', ref_inp.size)

    assert sdfg.apply_transformations(
        daceperiments.transforms.BasicRegisterCache,
        dict(array='buf'),
        validate=True) == 1
    sdfg.save('test.sdfg')

    compiled = sdfg.compile(optimizer=False)

    out = np.zeros_like(ref_out)
    buf = np.zeros_like(out, shape=(2, ))
    compiled(inp=ref_inp, out=out, buf=buf)

    np.testing.assert_allclose(out, ref_out)
