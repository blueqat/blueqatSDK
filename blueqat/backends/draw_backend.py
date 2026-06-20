# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backend for rendering quantum circuit diagrams using NetworkX and Matplotlib.
Fixed label overlapping on control nodes and adjusted vertical spacing for better readability.
"""

from .backendbase import Backend
from ..gate import *
import numpy as np
import math

class DrawCircuit(Backend):
    """Backend for draw output."""
    
    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        qlist = {}
        flg = 0
        time = 0
        add_edge = []
        remove_edge = []

        # 🛠️ 行間を開けるために ypos の間隔を 1.5 倍に広げる
        for i in range(n_qubits):
            qlist[i] = [{'num': flg, 'gate': 'q'+str(i), 'angle': '', 'xpos': 0, 'ypos': i * 1.5, 'type': 'qubit'}]
            flg += 1
        
        time += 1
        return gates, (qlist, n_qubits, [flg], [time], add_edge, remove_edge)

    def _postprocess_run(self, ctx):
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # カラーコードの定義
        color_gate = {}
        color_gate['X'] = color_gate['Y'] = color_gate['Z'] = '#0BB0E2'  # パウリ系: 水色
        color_gate['CX'] = color_gate['CY'] = color_gate['CZ'] = '#0BB0E2'
        color_gate['RX'] = color_gate['RY'] = color_gate['RZ'] = '#FCD000' # 回転系: 黄色
        color_gate['CRZ'] = color_gate['PHASE'] = '#FCD000'
        color_gate['H'] = color_gate['T'] = color_gate['S'] = '#E6000A'  # クリフォード系: 赤
        color_gate['SWAP'] = '#A020F0'                                   # スワップ: 紫
        color_gate['CCX'] = '#FF8C00'                                    # 3量子ビット系: オレンジ
        color_gate['M'] = 'white'
        
        qlist = ctx[0]
        n_qubits = ctx[1]
        flg = ctx[2][-1]
        time = ctx[3][-1]
        
        # 測定ゲートの位置
        m_xpos = max(8, time * 1.2)
        for i in range(n_qubits):
            # 🛠️ ypos を 1.5 倍ベースに合わせる
            qlist[i].append({'num': flg, 'gate': 'M', 'angle': '', 'xpos': m_xpos, 'ypos': i * 1.5 + math.floor((time-1)/30)*(n_qubits+1)*1.5, 'type': 'measurement'})
            flg += 1
        
        G = nx.Graph()

        for i in range(n_qubits):
            for j in range(len(qlist[i])-1):
                G.add_edge(qlist[i][j]['num'], qlist[i][j+1]['num'])
        
        # 2量子ビット以上の結合線の追加
        for item in ctx[4]:
            G.add_edge(item[0], item[1])

        for item in ctx[5]:
            G.remove_edge(item[0], item[1])

        # 🛠️ サイズの自動フィット（縦幅 height を少しゆったり広げる）
        width = max(8, m_xpos * 0.9)
        height = max(5, (n_qubits * 1.5 + 1) * 0.6)
        plt.figure(1, figsize=(width, height), dpi=100)

        labels = {}
        colors = {}
        angles = {}
        sizes = {}

        for i in range(n_qubits):
            for j in range(len(qlist[i])):
                angles[qlist[i][j]['num']] = qlist[i][j]['angle']
                
                gate_name = qlist[i][j]['gate']
                
                # 🛠️ 制御点（黒丸）の背後に文字が重なるバグを解消するため、空文字にする
                if gate_name == '' or gate_name == 'CZ' or gate_name == 'CRZ':
                    labels[qlist[i][j]['num']] = ''
                    colors[qlist[i][j]['num']] = 'black'
                    sizes[qlist[i][j]['num']] = 150  # 制御点の黒ドットサイズ
                elif qlist[i][j]['type'] == 'dummy':
                    labels[qlist[i][j]['num']] = ''
                    colors[qlist[i][j]['num']] = 'white'
                    sizes[qlist[i][j]['num']] = 0
                else:
                    labels[qlist[i][j]['num']] = gate_name
                    sizes[qlist[i][j]['num']] = 1000  # ゲートの丸を少しスマートに
                    if qlist[i][j]['type'] == 'qubit':
                        colors[qlist[i][j]['num']] = 'white'
                    else:
                        colors[qlist[i][j]['num']] = color_gate.get(gate_name, '#999999')

        # 座標の配置
        pos = {}
        for i in range(n_qubits):
            for j in range(len(qlist[i])):
                pos[qlist[i][j]['num']] = (qlist[i][j]['xpos'] * 1.2, (n_qubits*1.5+1)*(math.floor(time/30)+1) - qlist[i][j]['ypos'])

        # マージン用のダミーノード
        labels[flg]= ''
        colors[flg] = 'black'
        sizes[flg] = 0
        pos[flg] = (0, (n_qubits*1.5+1)*(math.floor(time/30)+1)+1)
        G.add_node(flg)
        labels[flg+1]= ''
        colors[flg+1] = 'black'
        sizes[flg+1] = 0
        pos[flg+1] = (0, 1)
        G.add_node(flg+1)
       
        nx.set_node_attributes(G, labels, 'label')
        nx.set_node_attributes(G, colors, 'color')
        nx.set_node_attributes(G, angles, 'angle')
        nx.set_node_attributes(G, sizes, 'size')

        options = {
            "font_size": 10,
            "edgecolors": "black",
            "linewidths": 1.5,
            "width": 1.5,
        }

        node_labels = nx.get_node_attributes(G, 'label')
        node_colors = [colors[i] for i in nx.get_node_attributes(G, 'color')]
        node_sizes = [sizes[i] for i in nx.get_node_attributes(G, 'size')]
        nx.draw_networkx(G, pos, labels = node_labels, node_color = node_colors, node_size = node_sizes, **options)

        # 角度（パラメータ）のテキスト描画
        pos_attrs = pos.copy()
        for i in pos_attrs:
            # 🛠️ 数値がゲートのすぐ下（かつ下の線に被らない位置）に来るよう y 座標のずらし量を -0.45 から -0.28 へ変更
            pos_attrs[i] = (pos_attrs[i][0] + 0.0, pos_attrs[i][1] - 0.28)
    
        node_attrs = nx.get_node_attributes(G, 'angle')
        custom_node_attrs = {}

        for node, attr in node_attrs.items():
            custom_node_attrs[node] = attr

        nx.draw_networkx_labels(G, pos_attrs, labels = custom_node_attrs, font_size=8)
        plt.show()
        return 

    def _one_qubit_gate_noargs(self, gate, ctx):
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + (math.floor(time/30)-1)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))
        
        time_adjust = time%30
        for idx in gate.target_iter(ctx[1]):
            ypos_adjust = idx * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
            qlist[idx].append({'num': flg, 'gate': gate.lowername.upper(), 'angle': '', 'xpos': time_adjust, 'ypos': ypos_adjust, 'type': 'gate'})
            flg += 1
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx

    gate_x = gate_y = gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = gate_tdg = _one_qubit_gate_noargs
    gate_s = gate_sdg = _one_qubit_gate_noargs
    
    def _one_qubit_gate_args_theta(self, gate, ctx):
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + (math.floor(time/30)-1)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))
        
        time_adjust = time%30
        for idx in gate.target_iter(ctx[1]):
            ypos_adjust = idx * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
            qlist[idx].append({'num': flg, 'gate': gate.lowername.upper(), 'angle': round(gate.theta, 2), 'xpos': time_adjust, 'ypos': ypos_adjust, 'type': 'gate'})
            flg += 1
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx

    gate_rx = gate_ry = gate_rz = _one_qubit_gate_args_theta
    gate_phase = _one_qubit_gate_args_theta

    def gate_i(self, gate, ctx):
        time = ctx[3][-1]
        ctx[3].append(time+1)
        return ctx
    
    def _two_qubit_gate_noargs(self, gate, ctx):
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        name = gate.lowername
        tg = 'X' if name == 'cx' else ('Y' if name == 'cy' else ('Z' if name == 'cz' else 'SWAP'))

        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + (math.floor(time/30)-1)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))

        time_adjust = time%30        
        for control, target in gate.control_target_iter(ctx[1]):
            qlist[target].append({'num': flg, 'gate': tg, 'angle': '', 'xpos': time_adjust, 'ypos': target * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
            flg += 1
            qlist[control].append({'num': flg, 'gate': 'CZ' if name=='cz' else '', 'angle': '', 'xpos': time_adjust, 'ypos': control * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
            flg += 1
            ctx[4].append((flg-2, flg-1))
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx
    
    gate_cx = gate_cy = gate_cz = gate_swap = _two_qubit_gate_noargs

    def _two_qubit_gate_args_theta(self, gate, ctx):
        """CRZゲートなどの描画"""
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]
        
        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + (math.floor(time/30)-1)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))

        time_adjust = time%30
        for control, target in gate.control_target_iter(ctx[1]):
            qlist[target].append({'num': flg, 'gate': 'RZ', 'angle': round(gate.theta, 2), 'xpos': time_adjust, 'ypos': target * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
            flg += 1
            qlist[control].append({'num': flg, 'gate': 'CRZ', 'angle': '', 'xpos': time_adjust, 'ypos': control * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
            flg += 1
            ctx[4].append((flg-2, flg-1))
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx

    gate_crz = _two_qubit_gate_args_theta

    def _three_qubit_gate_noargs(self, gate, ctx):
        """CCX(トフォリ)ゲートの描画"""
        flg = ctx[2][-1]
        time = ctx[3][-1]
        qlist = ctx[0]

        time_adjust = time%30
        if time_adjust == 0:
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + (math.floor(time/30)-1)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 30, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
            time += 1
            for i in range(ctx[1]):
                ypos_adjust = i * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5
                qlist[i].append({'num': flg, 'gate': '', 'angle': '', 'xpos': 0, 'ypos': ypos_adjust, 'type': 'dummy'})
                flg += 1
                ctx[5].append((flg-1, flg-1-ctx[1]))

        time_adjust = time%30
        c1, c2, target = gate.targets[0], gate.targets[1], gate.targets[2]
        
        qlist[target].append({'num': flg, 'gate': 'CCX', 'angle': '', 'xpos': time_adjust, 'ypos': target * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
        t_num = flg
        flg += 1
        
        qlist[c1].append({'num': flg, 'gate': '', 'angle': '', 'xpos': time_adjust, 'ypos': c1 * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
        c1_num = flg
        flg += 1
        
        qlist[c2].append({'num': flg, 'gate': '', 'angle': '', 'xpos': time_adjust, 'ypos': c2 * 1.5 + math.floor(time/30)*(ctx[1]+1)*1.5, 'type': 'gate'})
        c2_num = flg
        flg += 1
        
        ctx[4].append((t_num, c1_num))
        ctx[4].append((c1_num, c2_num))
        
        ctx[2].append(flg)
        ctx[3].append(time+1)
        return ctx

    gate_ccx = _three_qubit_gate_noargs
    gate_cswap = _three_qubit_gate_noargs

    def gate_measure(self, gate, ctx):
        return ctx

    gate_reset = _one_qubit_gate_noargs

    def run(self, gates, n_qubits, *args, **kwargs):
        """Blueqatのエントリーポイント"""
        gates, ctx = self._preprocess_run(gates, n_qubits, args, kwargs)
        for gate in gates:
            if hasattr(self, f"gate_{gate.lowername}"):
                ctx = getattr(self, f"gate_{gate.lowername}")(gate, ctx)
        self._postprocess_run(ctx)
        return ctx