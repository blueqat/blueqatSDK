import numpy as np

class FlexibleCircuitComposer:
    def __init__(self, max_block_size=2):
        """
        max_block_size: まとめる最大の量子ビット数 (2 = 最大4x4行列, 3 = 最大8x8行列)
        """
        self.max_block_size = max_block_size

    def compose(self, ops, n_qubits=None):
        """量子回路のゲート列（ops）をスキャンし、指定サイズ以下のブロックに集約する"""
        composed_blocks = []
        current_block = None
        
        for op in ops:
            gate_name = op.lowername.upper()
            
            # 🛠️ 【修正】 op.targets が int 型の場合はリストに包んでから set に変換する
            if isinstance(op.targets, int):
                gate_targets = {op.targets}
            else:
                gate_targets = set(op.targets)
            
            if current_block is None:
                current_block = {
                    'gates': [gate_name],
                    'targets': gate_targets
                }
                continue
            
            combined_targets = current_block['targets'].union(gate_targets)
            
            if len(combined_targets) <= self.max_block_size:
                current_block['gates'].append(gate_name)
                current_block['targets'] = combined_targets
            else:
                composed_blocks.append(current_block)
                current_block = {
                    'gates': [gate_name],
                    'targets': gate_targets
                }
                
        if current_block:
            composed_blocks.append(current_block)
            
        return composed_blocks