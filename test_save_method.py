#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ActorNetwork类中是否存在save_for_inference方法
"""

import os
import sys

# 确保当前目录在导入路径中
sys.path.insert(0, os.path.abspath('.'))

from networks import ActorNetwork

def test_actor_methods():
    """
    测试ActorNetwork类中的方法
    """
    print("检查ActorNetwork类的方法...")
    
    # 创建一个ActorNetwork实例
    actor = ActorNetwork(alpha=0.001, input_dims=10, fc1_dims=64, fc2_dims=64, 
                        n_actions=2, name="test_actor", chkpt_dir="./tmp")
    
    # 检查是否有save_for_inference方法
    has_method = hasattr(actor, 'save_for_inference')
    print(f"ActorNetwork是否有save_for_inference方法: {has_method}")
    
    # 列出所有方法
    methods = [method for method in dir(actor) if callable(getattr(actor, method)) and not method.startswith('_')]
    print(f"ActorNetwork的所有方法: {methods}")
    
    # 尝试调用方法
    if has_method:
        try:
            actor.save_for_inference()
            print("成功调用save_for_inference方法")
        except Exception as e:
            print(f"调用save_for_inference方法失败: {e}")

if __name__ == "__main__":
    test_actor_methods()