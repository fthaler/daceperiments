import copy
from collections import defaultdict

import dace
from dace import registry, symbolic
from dace.properties import Property, make_properties
from dace.sdfg import nodes, utils as sdutils
from dace.transformation.interstate.loop_detection import DetectLoop
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.pattern_matching import Transformation


@registry.autoregister
@make_properties
class BasicRegisterCache(Transformation):
    _before_state = dace.SDFGState()
    _loop_state = dace.SDFGState()
    _guard_state = dace.SDFGState()

    array = Property(dtype=str,
                     desc='Name of the array to replace by a register cache')

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


@registry.autoregister_params(singlestate=True)
class OnTheFlyMapFusion(Transformation):
    _first_map_entry = nodes.MapEntry(nodes.Map('', [], []))
    _first_tasklet = nodes.Tasklet('')
    _first_map_exit = nodes.MapExit(nodes.Map('', [], []))
    _array_access = nodes.AccessNode('')
    _second_map_entry = nodes.MapEntry(nodes.Map('', [], []))
    _second_tasklet = nodes.Tasklet('')

    @staticmethod
    def expressions():
        return [
            sdutils.node_path_graph(OnTheFlyMapFusion._first_map_entry,
                                    OnTheFlyMapFusion._first_tasklet,
                                    OnTheFlyMapFusion._first_map_exit,
                                    OnTheFlyMapFusion._array_access,
                                    OnTheFlyMapFusion._second_map_entry,
                                    OnTheFlyMapFusion._second_tasklet)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        first_map_entry = graph.node(
            candidate[OnTheFlyMapFusion._first_map_entry])
        first_tasklet = graph.node(candidate[OnTheFlyMapFusion._first_tasklet])
        first_map_exit = graph.node(
            candidate[OnTheFlyMapFusion._first_map_exit])
        array_access = graph.node(candidate[OnTheFlyMapFusion._array_access])

        # TODO: generalize to multiple tasklets/nodes inside first map
        first_map_nodes = (
            graph.all_nodes_between(first_map_entry, first_map_exit) -
            {first_map_entry})
        if first_map_nodes != {first_tasklet}:
            return False

        if len(first_tasklet.out_connectors) != 1:
            return False

        if (graph.in_degree(array_access) != 1
                or graph.out_degree(array_access) != 1):
            return False
        return True

    @staticmethod
    def _memlet_offsets(base_memlet, offset_memlet):
        """ Compute subset offset of `offset_memlet` relative to `base_memlet`.
        """
        def offset(base_range, offset_range):
            b0, e0, s0 = base_range
            b1, e1, s1 = offset_range
            assert e1 - e0 == b1 - b0 and s0 == s1
            return int(e1 - e0)

        return tuple(
            offset(b, o) for b, o in zip(base_memlet.subset.ranges,
                                         offset_memlet.subset.ranges))

    @staticmethod
    def _update_map_connectors(state, array_access, first_map_entry,
                               second_map_entry):
        """ Remove unused connector (of the to-be-replaced array) from second
            map entry, add new connectors to second map entry for the inputs
            used in the first map’s tasklets.
        """
        # Remove edges and connectors from arrays access to second map entry
        for edge in state.edges_between(array_access, second_map_entry):
            state.remove_edge_and_connectors(edge)
        state.remove_node(array_access)

        # Add new connectors to second map
        # TODO: implement for the general case with random naming
        for edge in state.in_edges(first_map_entry):
            if second_map_entry.add_in_connector(edge.dst_conn):
                state.add_edge(edge.src, edge.src_conn, second_map_entry,
                               edge.dst_conn, edge.data)

    @staticmethod
    def _read_offsets(state, array_name, first_map_exit, second_map_entry):
        """ Compute offsets of read accesses in second map.
        """
        # Get output memlet of first tasklet
        output_edges = state.in_edges(first_map_exit)
        assert len(output_edges) == 1
        write_memlet = output_edges[0].data

        # Find read offsets by looping over second map entry connectors
        offsets = defaultdict(list)
        for edge in state.out_edges(second_map_entry):
            if edge.data.data == array_name:
                second_map_entry.remove_out_connector(edge.src_conn)
                state.remove_edge(edge)
                offset = OnTheFlyMapFusion._memlet_offsets(
                    write_memlet, edge.data)
                offsets[offset].append(edge)

        return offsets

    def _replicate_tasklet(self, sdfg, array_access, first_tasklet,
                           first_map_exit, second_map_entry):
        """ Replicate tasklet of first map for reach read access in second map.
        """
        state = sdfg.node(self.state_id)
        array_name = array_access.data
        array = sdfg.arrays[array_name]

        read_offsets = self._read_offsets(state, array_name, first_map_exit,
                                          second_map_entry)

        # Replicate first tasklet once for each read offset access and connect
        # it to other tasklets accordingly
        for offset, edges in read_offsets.items():
            replicated_tasklet = copy.deepcopy(first_tasklet)
            tmp_name = sdfg.temp_data_name()
            sdfg.add_scalar(tmp_name, array.dtype, transient=True)
            tmp_access = state.add_access(tmp_name)
            state.add_edge(replicated_tasklet,
                           next(iter(replicated_tasklet.out_connectors)),
                           tmp_access, None, dace.Memlet(tmp_name))
            for input_edge in state.in_edges(first_tasklet):
                memlet = copy.deepcopy(input_edge.data)
                memlet.subset.offset(list(offset), negative=False)
                second_map_entry.add_out_connector(input_edge.src_conn)
                state.add_edge(second_map_entry, input_edge.src_conn,
                               replicated_tasklet, input_edge.dst_conn, memlet)
            for edge in edges:
                state.add_edge(tmp_access, None, edge.dst, edge.dst_conn,
                               dace.Memlet(tmp_name))

    def apply(self, sdfg: dace.SDFG):
        state = sdfg.node(self.state_id)
        first_map_entry = state.node(self.subgraph[self._first_map_entry])
        first_tasklet = state.node(self.subgraph[self._first_tasklet])
        first_map_exit = state.node(self.subgraph[self._first_map_exit])
        array_access = state.node(self.subgraph[self._array_access])
        second_map_entry = state.node(self.subgraph[self._second_map_entry])

        self._update_map_connectors(state, array_access, first_map_entry,
                                    second_map_entry)

        self._replicate_tasklet(sdfg, array_access, first_tasklet,
                                first_map_exit, second_map_entry)

        state.remove_nodes_from(
            [first_map_entry, first_tasklet, first_map_exit])
