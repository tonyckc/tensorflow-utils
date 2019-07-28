# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 19:06:19 2019

@author: ckc
"""
import frame_level_models
import video_level_models
def_model = 'LstmModel'

class A:
    def __init__(self):
        self.name = 'chen_ke_cheng'
    def method(self):
        print('method print')

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]# ？ for a in b 的用法
  return next(a for a in modules if a)# next 的用法 a for a in b if a? 的用法
#%%

Instance = A()
#如果Instance 对象中有属性name则打印self.name的值，否则打印'not find'
print(getattr(Instance, 'name', 'not find'))
print(getattr(Instance, 'age', 'not find'))
#如果有方法method，否则打印其地址，否则打印default
print(getattr(Instance, 'method', 'default'))
#如果有方法method，运行函数并打印None否则打印default
print(getattr(Instance, 'method', 'default')())


model = find_class_by_name(def_model,[frame_level_models,video_level_models])()#class实例化
print(model.create_model())