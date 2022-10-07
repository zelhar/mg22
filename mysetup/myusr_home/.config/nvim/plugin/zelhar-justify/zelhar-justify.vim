function ZelharJustify(width) range
    let g:my_first_line = a:firstline
    let g:my_last_line = a:lastline
    if a:width == ''
        let g:my_justify_width = 72
    else
        let g:my_justify_width = a:width
    endif
    py3file $HOME/.config/nvim/**/plugin/zelhar-justify/zelhar-justify.py
endfunction

function Get_range() range
    let g:my_first_line = a:firstline
    let g:my_last_line = a:lastline
endfunction

"example for function with range
function Count_words() range
  let lnum = a:firstline
  let n = 0
  while lnum <= a:lastline
	let n = n + len(split(getline(lnum)))
	let lnum = lnum + 1
  endwhile
  echo "found " . n . " words"
endfunction

command -nargs=1 ZelharJustify call ZelharJustify(<f-args>)
vnoremap <Leader>aq :call ZelharJustify(g:my_justify_width)<CR>
vnoremap <Leader>jt :call ZelharJustify(68)<CR>
"remove excessive spaces within a text.
vnoremap <Leader>ss :s/\s\s\+/\ /gc<CR>
"remove trailing white spaces.
vnoremap <Leader>ts :s/\s\+$//gc<CR>

"default line width
let g:my_justify_width = 72
