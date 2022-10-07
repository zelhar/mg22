#!/usr/bin/python

import os
import sys
import re
import textwrap
import vim

def prepare_string(s):
#eliminate possible useless invisible character
    temp_str = s.replace(u'\ufeff', '')
    return temp_str

def find_next_paragraph(s):
    return s.index('\n\n')

def process_paragraph(par, width=72, action='ljust'):
    pass

class Justify_Paragraph:
    """Test Justification Given an array of words and a length L, format the
    text such that each line has exactly L characters and is fully (left and
    right) justified.  
    
    You should pack your words in a greedy approach; that is, pack as many words
    as you can in each line. 
    
    Pad extra spaces ' ' when necessary so that each line has exactly L
    characters.  
    
    Extra spaces between words should be distributed as evenly as possible. If
    the number of spaces on a line do not divide evenly between words, the empty
    slots on the left will be assigned more spaces than the slots on the right.  
    
    For the last line of text, it should be left justified and no extra space is
    inserted between words.  
    
    For example, words: ["This", "is", "an",
    "example", "of", "text", "justification."] L: 16.  Return the formatted lines
    as:
    [
    "This    is    an",
    "example  of text",
    "justification.  "
    ]

    A line other than the last line might contain only one word. What should you do in this case?
    In this case, that line should be left-justified.

    If a word exceeds length L, it is placed unbroken in its own line.
    """
    def get_next_line(self, words, L):
        """input is words - a list of words, and L - the max length of a line.
        Result returned is the index i so that words[0:i] is the maximalline
        which doesn't exceed L (except for the case of a single word of length
        greater then L)."""
        my_length = len(words[0])
        i = 0
        next_length = my_length
        while (my_length <= L):
            i+=1
            next_length = my_length + len(words[i]) + 1
            if (next_length <= L):
                my_length = next_length
        return i #words[0:i] is the line that meats length specifications

    def divide_into_lines(self, words, L):
        """gets a list of words and line length L as input. Returns a list of
        lines that contains all the words in 'words' so that each line doesn't
        exceed the length L (except the single long word case)."""
        my_lines = []
        i = 0
        while (i < len(words)):
            k = self.get_next_line(words[i:], L)
            #my_lines[i] = ' '.join(words[i:k])
            my_lines[i] = words[i:k]
            i = k
        return my_lines
    def justufy_line(self, wordsline, L):
        """Gets as input list of words in one line. Return a string containing
        the words in a line of exactly length L and with spaces distributed
        evenly aka justify."""
        
        num_of_spaces = len(wordsline) - 1.
        this_line = ' '.join(wordsline)
        line_length = len(this_line)
        if (line_length >= L):
            return this_line
        extra_space_to_fill = L - line_length
        total_spaces_length = num_of_spaces + extra_space_to_fill
        x = total_spaces_length // num_of_spaces
        y = total_spaces_length % num_of_spaces
        my_spaces_list = [' '*x]*num_of_spaces
        #we need to distribut now extra y<num_of_spaces spaces as evenly as
        #possible.
        n = num_of_spaces
        m = y
        a = n // m
        b = n % m
        #so n = am + b, b<m
        temp_model = [1]*m + [0]*(n-m)



def distribute(balls_num, slots_num):
    """distribute balls_num (int) over slots_num (int) as evenly as possibly. Returs a
    list of length slots, each element is an integer that represents the number
    of balls in the slot"""
    a = balls_num // slots_num
    b = balls_num % slots_num
    slot_list = [a]*slots_num
    left = 0
    right = slots_num - 1
    mid_right = slots_num // 2
    mid_left = mid_right - 1
    while (b > 0):
        slot_list[mid_left] += 1
        mid_left -= 1
        b -= 1
        if (b == 0):
            break
        slot_list[mid_right] += 1
        mid_right += 1
        b -= 1
        if (b == 0):
            break
        slot_list[left] += 1
        left += 1
        b -= 1
        if (b == 0):
            break
        slot_list[right] += 1
        right -= 1
        b -= 1
        if (b == 0):
            break
    return slot_list

def split_into_paragraphs(text, sep='\n\n'):
    if (type(text) != str) | (type(sep) != str):
        return []
    my_list = text.split(sep)
    return my_list

def justify_line(text, width=72):
    """Justifies the line into specified width. In case the line already exceeds
    width it just returns the same line. Also spaces will be normalized,
    trailing space or excess space between words will be removes"""
    my_list = text.split()
    num_words = len(my_list)
    text_line = ' '.join(my_list)
    line_length = len(text_line) #length of spaces in the sentence
    nonspace_text_length = line_length - num_words + 1
    if (line_length >= width):
        return text_line
    slots = num_words - 1
    balls = width - nonspace_text_length
    slot_list = distribute(balls, slots)
    my_line = my_list[0]
    for i in range(len(slot_list)):
        my_line += ' '*slot_list[i]
        my_line += my_list[i+1]
    return my_line


def indentation(s, tabsize=4):
    sx = s.expandtabs(tabsize)
    if sx.isspace():
        return 0
    else:
        return len(sx) - len(sx.lstrip())


def justify_paragraph(text, width=72):
    #now adding indentation check
    indent_level = indentation(text, 4)
    my_text = text.strip()
    my_lines = textwrap.wrap(my_text, width)
    for i in range(len(my_lines)-1):
        my_lines[i] = ' '*indent_level + justify_line(my_lines[i], width)
    if len(my_lines) > 1:
        my_lines[-1] = ' '*indent_level + my_lines[-1]
    return "\n".join(my_lines)

def justify_text(text, width=72, sep='\n\n'):
    my_pars = split_into_paragraphs(text, sep)
    for i in range(len(my_pars)):
        if (len(my_pars[i]) >0):
            my_pars[i] = justify_paragraph(my_pars[i], width)
    just_text = "\n\n".join(my_pars)
    return just_text

my_range = vim.current.range

if (__name__ == '__main__'):
    #print("my name is " + __name__)
    #print("range is: " + str(my_range.start) + " to " + str(my_range.end))
    #print('Number of arguments:', len(sys.argv), 'arguments.')
    #print('Argument List:', str(sys.argv))
    if (len(sys.argv) > 1):
        my_just_text = justify_text(str(sys.argv))
        #print(my_just_text)
else:
    pass
    #print(__name__)
    #print("range is: " + my_range.start + " to " + my_range.end)

range_start = int(vim.eval("g:my_first_line"))
range_end = int(vim.eval("g:my_last_line"))
try:
    width = int(vim.eval("g:my_justify_width"))
except:
    width = 72
#print("the range is: " + str(range_start) + " to " + str(range_end))
buffer_selected_lines = vim.current.buffer[range_start - 1:range_end]
text = "\n".join(buffer_selected_lines)
#print("the content of the selected range is:\n" + text)
j_text = justify_text(text, width)
b = vim.current.buffer


start_index = range_start - 1
end_index = range_end -1
num_lines = range_end - start_index
if (b[start_index] == ''):
    start_index +=1
if (b[range_end -1] == ''):
    end_index -=1
#b[start_index : range_end] = None #delete the selected text to replace it
j_list = j_text.split('\n') 
#b.append(j_list, range_start - 2)
#print(j_text)
#print("first line is: \n" + j_list[0])
#b.append(j_list[0], range_start - 1)
#print("last line is: \n" + j_list[-1])
#b.append(j_list, end_index+1)
#b[start_index : end_index + 1] = None #delete the selected text to replace it
b.append(j_list, start_index)
shift = len(j_list)
b[start_index + shift: end_index + 1 + shift] = None #delete the selected text to replace it
