## My 2 cent suggestion for fixing the inferiority of indentation based syntax

Python syntax is very adequate in 99% of the time. Indentation makes code
legible and that's why C-syntax or Lisp-syntax use informally too.
The obvious problem is what to do when the indentation becomes too deep. Even if
you break language stamdard and employs 2-space indentation it just delays
dealing with the actual issue.

I have a symple suggestion: add special literals that indicate indentation level
and can be replace indentation. Specifically I suggest the following literals
(not necessarily these characters but what they should stand for:

First, just include a delimeter for starting and ending a c-style delimiter.
Just make it obscure so it souldn't waste vital keyboard real estate.
I suggest for example `»{` to start c-style block. After it indentation level doesn't 
switches back to 0 after '\n', you have to explicitly end it.

A nother token, I suggest `»`, indicates '\n\t' if the line prefix it ends
contains any non-space valid code piece, in other words if in normal syntax you
would go '\n\t', in the alternative syntax you would type ' » ' and remain on
the same line. A nomber before indicates indent level (1 is implied if there is
none), so ` 3» ` is like '\n\t\t' but you stay physically on the same line.
If `»` appear as the first non-space literal on a line, than it functions
logically the same as before but without the preceiding '\n' which would be
illegal. So if your line starts with: '^\t foo(x)' (tab space space) then it is
equivalent to '^»  foo(x)' The space before '»' doesn't count as space if before
it there is a non-space, the
first space character after '»' also is just for visual separation and doesn't
count as extra space for indent sake,

As long as you are on a line, » and « designate increasing and decreasing the
indentation level. When you end the line with `\n', indentation is back to 0
like normal syntax. You then use the normal python indentation convention or 
augment it with 2», » etc.

  * `»` : `\nTab`, optionally with a number indicating depth 
    `2»` indicates line breaks and 2 tabs if it comes midline, 
    or just 2 tabs if it is the firts character in the line. Defauly is `»` = `1»`.
  * `«` : The inverse operation.
  * `»{` : Block mode that is sticky, meanining the indent level remains the
    same regardless of white space and line breaks until you invoke the inverse:
  * `«}` : end current block.
  * `;` : used midline to indicate multiple statements becomes much more usefule
    now.
  * `\n...` : indicates continuing the last line (visual break, remaining on
  same line logically.

Yes it's true, `»{` is exactly the C style '{'...
just with a more obscure token that doesn't waste much in terms of losing
quickly accessible keys.


```
class SomeImportantClass():
    """
    very classy class with deep indentations.
    """
    ...
    # taken from Python reference.
    # Compute the list of all permutations of l
    def perm(l):
        if len(l) <= 1:
            return [l]
        r = []
        for i in range(len(l)):
            s = l[:i] + l[i+1:]
            p = perm(s)
                for x in p:
                    r.append(l[i:i+1] + x)
        return r

# alternative:

class SomeImportantClass():
    """
    very classy class with shallow visual indentations.
    but same logical one.
    """
    ...
    # taken from Python reference.
    # Compute the list of all permutations of l
    def perm(l): » if len(l) <= 1: » return [l] « r = []
    » for i in range(len(l)):
    »» s = l[:i] + l[i+1:]; p = perm(s);
    3» for x in p: » r.append(l[i:i+1] + x) « return r
    # in the above line, the 'for' starts exactly on indent lever '\t\t\t\t'.
    # the indentation level before '3' all spaces count
    # But midline, 'r.append..' starts exatly at: linebreak + current indent
    # level + extra tab.
    # return comes after one inverse tab so exactly one tab level less

```

I think the '«','»' suggestion remains pythonic in the sense that you don't need
to delimit a block from both sides. In python you always 'enter' into a block 
Implicitly every line of code first needs to be entered into its proper indent
level. You never exit a block in python, at least not at the new statement
level. If blocs exist in python, it is mid-statement.
Anyway these new char are just another way of expressing how to enter into a
specific indentation level just more condensely without necessitating line
breaks and a ton of space.
