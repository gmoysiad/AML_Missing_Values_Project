# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:57:06 2020

@author: s.filippou
"""

class Node(object):

    def __init__ (self,d):
        self.data = d
        self.next_node = None

    def get_next (self):
        return self.next_node

    def set_next (self, n):
        self.next_node = n

    def get_data (self):
        return self.data

    def set_data (self, d):
        self.data = d
        
        
class LinkedList(object):

    def __init__(self, r = None):
        self.head = None
        self.origine = None

    def get_size (self):
        return self.size

    def push (self, d):
        try :
            next_data = self.head.data
            new_node = Node(d + next_data)
        except:
            new_node = Node(d)
        new_node.next = self.head
        self.head = new_node
        
    def deleteNode(self, position): 
  
        # If linked list is empty 
        if self.head == None: 
            return 
  
        # Store head node 
        temp = self.head 
  
        # If head needs to be removed 
        if position == 0: 
            self.head = temp.next
            temp = None
            return 
  
        # Find previous node of the node to be deleted 
        for i in range(position -1 ): 
            temp = temp.next
            if temp is None: 
                break
  
        # If position is more than number of nodes 
        if temp is None: 
            return 
        if temp.next is None: 
            return 
  
        # Node temp.next is the node to be deleted 
        # store pointer to the next of node to be deleted 
        next = temp.next.next
  
        # Unlink the node from linked list 
        temp.next = None
  
        temp.next = next 

    # def remove (self, d):
    #     this_node = self.root
    #     prev_node = None

    #     while this_node:
    #         if this_node.get_data() == d:
    #             if prev_node:
    #                 prev_node.set_next(this_node.get_next())
    #             else:
    #                 self.root = this_node.get_next()
    #             self.size -= 1
    #             return True		# data removed
    #         else:
    #             prev_node = this_node
    #             this_node = this_node.get_next()
    #     return False  # data not found

    # def find (self, d):
    #     this_node = self.root
    #     while this_node:
    #         if this_node.get_data() == d:
    #             return d
    #         else:
    #             this_node = this_node.get_next()
    #     return None        