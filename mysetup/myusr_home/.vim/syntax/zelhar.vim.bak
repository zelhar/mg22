" Zelhar MathJax extension to markdown syntax

if exists("b:current_syntax")
  finish
endif

"Load the standard markdown
"runtime! syntax/ghmarkdown.vim

if !exists('main_syntax')
  let main_syntax = 'zelhar'
endif

runtime! syntax/zelhar-ghmarkdown.vim
unlet! b:current_syntax

"My additions



" MY MathML/MathJax addition
let terrible="\(\s\|\n\)\zs\$\{1}[^ $]\{1}[^$]\{-}[^ $]\$\ze\(\s\|\n\)"
syn match       asciimathml                 /¡¡\_[^¡]\{-}¡¡/
syn match       asciimathml2                /%%\_[^%]\_.\{-}%%/
syn match 		displayedMathML 	    	/\(\$\{2}\)\_[^$]\+\1/
"syn match 		displayedMathML     /\(\s\|\n\)\zs\$\{2}[^ $]\{1}[^$]\{-}[^ $]\$\{2}\ze\(\s\|\n\)/
"syn match 		displayedMathML     /\(\s\|\n\)\zs\$\{2}[^ $]\{1}[^$]\{-}[^ $]\$\{2}\ze\(\_[^$]\)/
syn match       displayedMathML2            /\\\[\_.\{-}\\\]/ 
syn match       inlinemath                  /\\(\_.\{-}\\)/ 
"syn match 		inlinemath2			    	/\(\$\)\_[^$]\+\1/
syn match 		inlinemath2	/\([,.()\n\t ]\|^\s*\)\zs\$\_[^$]\+\$/
"syn match 		inlinemath3	/^\s\{0,1}\$\_[^$]\+\$/
"syn match 		inlinemath2	/\(\s*\|\n\)\zs\$\{1}[^ $]\{1}\_[^$]\{-}[^ $]\$\ze\(\s\|\n\|\.\|\,\)/
"syn match 		inlinemath2	/\(\s\|\n\)\zs\$\{1}[^ $]\{1}[^$]\{-}[^ $]\$\ze\(\_[^$]\)/
"syn match 		inlinemath2	/\$\{1}[^$]\{-}\$\{1}/
syn match       trailingspace               /\s\{2,}$/ 

"My MathJax Highlighting for the groups
hi def link asciimathml             Operator
hi def link asciimathml2            Operator
hi def link displayedMathML         Typedef 
hi def link displayedMathML2        Typedef 
hi def link inlinemath              Operator
hi def link inlinemath2             Operator
hi def link inlinemath3             Operator
hi def link mkdMacro                mkdURL
hi def link trailingspace           ErrorMsg 


let b:current_syntax = "zelhar"
if main_syntax ==# 'zelhar'
  unlet main_syntax
endif
