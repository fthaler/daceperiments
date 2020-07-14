import dace
from dace import registry, symbolic
from dace.sdfg import nodes
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.pattern_matching import Transformation


@registry.autoregister
class BasicRegisterCache(Transformation):
    _before_state = dace.SDFGState()
    _loop_state = dace.SDFGState()
    _guard_state = dace.SDFGState()

    @staticmethod
    def expressions():
        sdfg = dace.SDFG('_')
        before_state, loop_state, guard_state = (
            BasicRegisterCache._before_state, BasicRegisterCache._loop_state,
            BasicRegisterCache._guard_state)
        sdfg.add_nodes_from((before_state, loop_state, guard_state))
        sdfg.add_edge(before_state, guard_state, dace.InterstateEdge())
        sdfg.add_edge(guard_state, loop_state, dace.InterstateEdge())
        sdfg.add_edge(loop_state, guard_state, dace.InterstateEdge())
        return [sdfg]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    def _buffer_memlets(self, states):
        for state in states:
            for edge in state.edges():
                src, dst = edge.src, edge.dst
                if (isinstance(src, nodes.AccessNode) and src.data == 'buf'
                        or isinstance(dst, nodes.AccessNode)
                        and dst.data == 'buf'):
                    yield edge.data

    def _get_offsets(self, states):
        min_offset, max_offset = 1000, -1000
        for memlet in self._buffer_memlets(states):
            rb, re, _ = memlet.subset.ranges[0]
            rb_offset = rb - symbolic.symbol('i')
            re_offset = re - symbolic.symbol('i')
            min_offset = min(min_offset, rb_offset, re_offset)
            max_offset = max(max_offset, rb_offset, re_offset)
        return min_offset, max_offset

    def _make_indices_absolute(self, states, min_offset):
        for memlet in self._buffer_memlets(states):
            memlet.subset.replace({symbolic.symbol('i'): -min_offset})

    def _insert_shift_after(self, sdfg, state, min_offset, max_offset):
        shift_state = sdfg.add_state_after(state)
        buf_read = shift_state.add_read(self.array)
        buf_write = shift_state.add_write(self.array)

        shift_state.add_edge(
            buf_read, None, buf_write, None,
            dace.Memlet.simple(self.array,
                subset_str=f'1:{max_offset - min_offset + 1}',
                other_subset_str=f'0:{max_offset - min_offset}'))

    def apply(self, sdfg: dace.SDFG):
        before_state = sdfg.node(self.subgraph[self._before_state])
        loop_state = sdfg.node(self.subgraph[self._loop_state])
        guard_state = sdfg.node(self.subgraph[self._guard_state])

        min_offset, max_offset = self._get_offsets([before_state, loop_state])
        self._make_indices_absolute([before_state, loop_state], min_offset)

        sdfg.arrays[self.array].shape = (max_offset - min_offset + 1,)

        self._insert_shift_after(sdfg, before_state, min_offset, max_offset)
        self._insert_shift_after(sdfg, loop_state, min_offset, max_offset)
