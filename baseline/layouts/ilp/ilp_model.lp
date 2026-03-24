\ Model 2026-03-24_21-37-40
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  flow_source_0 + flow_source_1 + flow_source_2
Subject To
 hold_constraint_0: hold_0_1 + hold_0_2 + hold_0_3 + hold_0_4 + hold_0_5
   + hold_0_6 + hold_0_7 + hold_0_8 + hold_0_9 + hold_0_10 + hold_0_11
   + hold_0_12 = 1
 end_constraint_0: start_0 + hold_0_1 + 2 hold_0_2 + 3 hold_0_3
   + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7 + 8 hold_0_8
   + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12 <= 80
 hold_constraint_1: hold_1_1 + hold_1_2 + hold_1_3 + hold_1_4 + hold_1_5
   + hold_1_6 + hold_1_7 + hold_1_8 + hold_1_9 = 1
 end_constraint_1: start_1 + hold_1_1 + 2 hold_1_2 + 3 hold_1_3
   + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7 + 8 hold_1_8
   + 9 hold_1_9 <= 80
 hold_constraint_2: hold_2_1 + hold_2_2 + hold_2_3 + hold_2_4 + hold_2_5
   + hold_2_6 + hold_2_7 + hold_2_8 + hold_2_9 = 1
 end_constraint_2: start_2 + hold_2_1 + 2 hold_2_2 + 3 hold_2_3
   + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7 + 8 hold_2_8
   + 9 hold_2_9 <= 80
 node_flow_constraint_0: flow_source_0 - flow_0_sink - flow_0_1 + flow_1_0
   - flow_0_2 + flow_2_0 = 0
 node_flow_constraint_1: flow_source_1 - flow_1_sink + flow_0_1 - flow_1_0
   - flow_1_2 + flow_2_1 = 0
 node_flow_constraint_2: flow_source_2 - flow_2_sink + flow_0_2 - flow_2_0
   + flow_1_2 - flow_2_1 = 0
 node_throughput_constraint_0: - 62121.15085079249 hold_0_1
   - 29858.150496086 hold_0_2 - 19221.81833262335 hold_0_3
   - 13844.04979262387 hold_0_4 - 10544.15612047309 hold_0_5
   - 8327.078718310791 hold_0_6 - 6796.581192589725 hold_0_7
   - 5689.19059161553 hold_0_8 - 4802.854179301406 hold_0_9
   - 3988.257260149664 hold_0_10 - 3308.0343535219 hold_0_11
   - 2812.089310406632 hold_0_12 + flow_source_0 + flow_1_0 + flow_2_0
   <= 0
 node_throughput_constraint_1: - 42487.98909430546 hold_1_1
   - 20792.87034781813 hold_1_2 - 12941.006561159 hold_1_3
   - 8852.023483320118 hold_1_4 - 6594.060567713608 hold_1_5
   - 4894.543803077809 hold_1_6 - 3987.448844655458 hold_1_7
   - 2707.913433154988 hold_1_8 - 1753.347438117242 hold_1_9
   + flow_source_1 + flow_0_1 + flow_2_1 <= 0
 node_throughput_constraint_2: - 42487.98909430546 hold_2_1
   - 20792.87034781813 hold_2_2 - 12941.006561159 hold_2_3
   - 8852.023483320118 hold_2_4 - 6594.060567713608 hold_2_5
   - 4894.543803077809 hold_2_6 - 3987.448844655458 hold_2_7
   - 2707.913433154988 hold_2_8 - 1753.347438117242 hold_2_9
   + flow_source_2 + flow_0_2 + flow_1_2 <= 0
 edge_disable_constr_source_0: start_0 + 80 switch_source_0 <= 80
 edge_disable_constr_source_1: start_1 + 80 switch_source_1 <= 80
 edge_disable_constr_source_2: start_2 + 80 switch_source_2 <= 80
 edge_disable_constr_0_sink: - start_0 - hold_0_1 - 2 hold_0_2 - 3 hold_0_3
   - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7 - 8 hold_0_8
   - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   + 80 switch_0_sink <= 0
 edge_disable_constr_1_sink: - start_1 - hold_1_1 - 2 hold_1_2 - 3 hold_1_3
   - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7 - 8 hold_1_8
   - 9 hold_1_9 + 80 switch_1_sink <= 0
 edge_disable_constr_2_sink: - start_2 - hold_2_1 - 2 hold_2_2 - 3 hold_2_3
   - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7 - 8 hold_2_8
   - 9 hold_2_9 + 80 switch_2_sink <= 0
 edge_disable_constr1_0_1: start_0 - start_1 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 80 switch_0_1 <= 80
 edge_disable_constr2_0_1: - start_0 + start_1 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   + 80 switch_0_1 <= 80
 edge_disable_constr1_1_0: - start_0 + start_1 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 80 switch_1_0 <= 80
 edge_disable_constr2_1_0: start_0 - start_1 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 + 80 switch_1_0 <= 80
 edge_disable_constr1_0_2: start_0 - start_2 + hold_0_1 + 2 hold_0_2
   + 3 hold_0_3 + 4 hold_0_4 + 5 hold_0_5 + 6 hold_0_6 + 7 hold_0_7
   + 8 hold_0_8 + 9 hold_0_9 + 10 hold_0_10 + 11 hold_0_11 + 12 hold_0_12
   + 80 switch_0_2 <= 80
 edge_disable_constr2_0_2: - start_0 + start_2 - hold_0_1 - 2 hold_0_2
   - 3 hold_0_3 - 4 hold_0_4 - 5 hold_0_5 - 6 hold_0_6 - 7 hold_0_7
   - 8 hold_0_8 - 9 hold_0_9 - 10 hold_0_10 - 11 hold_0_11 - 12 hold_0_12
   + 80 switch_0_2 <= 80
 edge_disable_constr1_2_0: - start_0 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 80 switch_2_0 <= 80
 edge_disable_constr2_2_0: start_0 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 + 80 switch_2_0 <= 80
 edge_disable_constr1_1_2: start_1 - start_2 + hold_1_1 + 2 hold_1_2
   + 3 hold_1_3 + 4 hold_1_4 + 5 hold_1_5 + 6 hold_1_6 + 7 hold_1_7
   + 8 hold_1_8 + 9 hold_1_9 + 80 switch_1_2 <= 80
 edge_disable_constr2_1_2: - start_1 + start_2 - hold_1_1 - 2 hold_1_2
   - 3 hold_1_3 - 4 hold_1_4 - 5 hold_1_5 - 6 hold_1_6 - 7 hold_1_7
   - 8 hold_1_8 - 9 hold_1_9 + 80 switch_1_2 <= 80
 edge_disable_constr1_2_1: - start_1 + start_2 + hold_2_1 + 2 hold_2_2
   + 3 hold_2_3 + 4 hold_2_4 + 5 hold_2_5 + 6 hold_2_6 + 7 hold_2_7
   + 8 hold_2_8 + 9 hold_2_9 + 80 switch_2_1 <= 80
 edge_disable_constr2_2_1: start_1 - start_2 - hold_2_1 - 2 hold_2_2
   - 3 hold_2_3 - 4 hold_2_4 - 5 hold_2_5 - 6 hold_2_6 - 7 hold_2_7
   - 8 hold_2_8 - 9 hold_2_9 + 80 switch_2_1 <= 80
 edge_flow_constr_source_0: flow_source_0 - 7.1e+06 switch_source_0 <= 0
 edge_flow_constr_source_1: flow_source_1 - 6.05e+06 switch_source_1 <= 0
 edge_flow_constr_source_2: flow_source_2 - 6.3e+06 switch_source_2 <= 0
 edge_flow_constr_0_sink: flow_0_sink - 6.95e+06 switch_0_sink <= 0
 edge_flow_constr_1_sink: flow_1_sink - 6.2e+06 switch_1_sink <= 0
 edge_flow_constr_2_sink: flow_2_sink - 7.25e+06 switch_2_sink <= 0
 edge_flow_constr_0_1: flow_0_1 - 695.80078125 switch_0_1 <= 0
 edge_flow_constr_1_0: flow_1_0 - 695.80078125 switch_1_0 <= 0
 edge_flow_constr_0_2: flow_0_2 - 799.560546875 switch_0_2 <= 0
 edge_flow_constr_2_0: flow_2_0 - 799.560546875 switch_2_0 <= 0
 edge_flow_constr_1_2: flow_1_2 - 885.009765625 switch_1_2 <= 0
 edge_flow_constr_2_1: flow_2_1 - 885.009765625 switch_2_1 <= 0
Bounds
Binaries
 hold_0_1 hold_0_2 hold_0_3 hold_0_4 hold_0_5 hold_0_6 hold_0_7 hold_0_8
 hold_0_9 hold_0_10 hold_0_11 hold_0_12 hold_1_1 hold_1_2 hold_1_3 hold_1_4
 hold_1_5 hold_1_6 hold_1_7 hold_1_8 hold_1_9 hold_2_1 hold_2_2 hold_2_3
 hold_2_4 hold_2_5 hold_2_6 hold_2_7 hold_2_8 hold_2_9 switch_source_0
 switch_source_1 switch_source_2 switch_0_sink switch_1_sink switch_2_sink
 switch_0_1 switch_1_0 switch_0_2 switch_2_0 switch_1_2 switch_2_1
Generals
 start_0 start_1 start_2
End
