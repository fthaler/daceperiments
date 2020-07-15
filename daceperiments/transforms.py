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
                if (isinstance(src, nodes.AccessNode)
                        and src.data == self.array
                        or isinstance(dst, nodes.AccessNode)
                        and dst.data == self.array):
                    yield edge.data

    def _get_loop_axis(self, loop_state, loop_var):
        def contains_loop_var(subset_range):
            return any(loop_var in {s.name
                                    for s in r.free_symbols}
                       for r in subset_range)

        for memlet in self._buffer_memlets([loop_state]):
            return [contains_loop_var(r)
                    for r in memlet.subset.ranges].index(True)

    def _get_buffer_size(self, state, loop_var, loop_axis):
        min_offset, max_offset = 1000, -1000
        for memlet in self._buffer_memlets([state]):
            rb, re, _ = memlet.subset.ranges[loop_axis]
            rb_offset = rb - symbolic.symbol(loop_var)
            re_offset = re - symbolic.symbol(loop_var)
            min_offset = min(min_offset, rb_offset, re_offset)
            max_offset = max(max_offset, rb_offset, re_offset)
        return max_offset - min_offset + 1

    def _replace_indices(self, states, loop_var, loop_axis, buffer_size):
        for memlet in self._buffer_memlets(states):
            rb, re, rs = memlet.subset.ranges[loop_axis]
            memlet.subset.ranges[loop_axis] = (rb % buffer_size,
                                               re % buffer_size, rs)

    def apply(self, sdfg: dace.SDFG):
        before_state = sdfg.node(self.subgraph[self._before_state])
        loop_state = sdfg.node(self.subgraph[self._loop_state])
        guard_state = sdfg.node(self.subgraph[self._guard_state])
        loop_var = next(iter(sdfg.in_edges(guard_state)[0].data.assignments))

        loop_axis = self._get_loop_axis(loop_state, loop_var)

        buffer_size = self._get_buffer_size(loop_state, loop_var, loop_axis)
        self._replace_indices(sdfg.states(), loop_var, loop_axis, buffer_size)

        array = sdfg.arrays[self.array]
        # TODO: generalize
        if array.shape[loop_axis] == array.total_size:
            array.shape = tuple(buffer_size if i == loop_axis else s
                                for i, s in enumerate(array.shape))
            array.total_size = buffer_size
