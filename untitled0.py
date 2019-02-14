#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 22:40:12 2019

@author: mbzhao
"""
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Soluation :
    def rm(self, head, val):
        
        while head != None and head.val == val:
            head = head.next
        
        pre = head
        nex = pre.next
        while nex is not None:
            if nex.val == val:
                pre.next = nex.next
                nex = nex.next
            else:
                pre = nex
                nex = nex.next
        return head
    
if __name__ == '__main__':
    
    l = [6,2,6,3,6,2]
    H = ListNode(None)
    F = H
    for v in l:
        Next = ListNode(v)
        F.next = Next
        F = Next
    
    
    solver = Soluation()
    
    re = solver.rm(H.next, 6)