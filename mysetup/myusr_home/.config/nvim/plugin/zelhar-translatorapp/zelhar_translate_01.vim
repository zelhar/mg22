function! TranslateCurrentWord(inputword)
    if g:toggle_translate_mode == 0
        echo "error, translate mode is disabled"
        return 0
    else
        let this_buffer = bufnr('%')
        let temp_buffer = g:dump_buffer_nr
        let temp_bufwinnr = bufwinnr(temp_buffer)
        let this_bufwinnr = bufwinnr(this_buffer)
        if mode() == 'n'
            execute "normal"."\"1yw"
        endif
        if a:inputword == ''
            let searchword = @1
        else    
            let searchword = a:inputword
        endif
        "execute "b".temp_buffer
        execute temp_bufwinnr."wincmd w" 
        execute "%d"
        "execute "goto 1"
        "execute "silent read !dict -d fd-deu-eng ".searchword
        "I replaced dict with grep below:
        "let myline = "silent 0read !grep -w -i -h -m4 -A1 '^".searchword."' "
        let myline = "silent 0read !grep -w -i -h '^".searchword."' "
        "let myline .= g:plugin_dir.'/deu-eng.dict'
        let myline .= g:dict_file
        execute myline
        execute "goto 1"
        execute "match Error /".searchword."\\c/" 
        "execute "b".this_buffer
        execute this_bufwinnr."wincmd w" 
        "execute "silent !dict -d fd-deu-eng <cword> > ~/tempfile.txt" 
    endif
endfunction

"default is for nvim
let g:plugin_dir = expand($HOME).'/.config/nvim/plugin/zelhar-translatorapp'

"set default dictionary language
let g:dict_language = 'deu-eng.dict'

"default dictionary file:
let g:dict_file = g:plugin_dir.'/'.g:dict_language

function! GrepCurrentWord()
    let my_line = "!grep -w -h -i -m4 -A1 '^<cword>' "
    "let my_line .= g:plugin_dir.'/deu-eng.dict'
    let my_line .= g:dict_file
    let my_line .= " > ~/tempfile.txt"
    execute 'silent '.my_line
endfunction

command TranslateCurrentWord call TranslateCurrentWord('')

command GrepCurrentWord call GrepCurrentWord()

"I've moved all map commands inside the Toggle function to be toggled with the
"mode
"nnoremap <Leader>t :silent TranslateCurrentWord<Cr>
"            \:echo TranslateCurrentWord()<Cr>
"            \:!cat ~/tempfile.txt<Cr>
"            \:TranslateCurrentWord<Cr> 

"vnoremap <Leader>t "1y:silent call TranslateCurrentWord(@1)<Cr>

"nnoremap <Leader>y :silent GrepCurrentWord<Cr>

let g:toggle_translate_mode = 0
"dump buffer still doesn't exists so set impossible number
let g:dump_buffer_nr = -1

"remember which buffer we are working at before starting to mess aroung
let g:working_buffer_nr = bufnr('%')

"a temp file is no longer used to save writes into flash drive. Instead a
"temporary buffer is used without any file and no writes to storage device.
let g:tempfile = "~/tempfile.txt"
"let tempfile = tempname() 

function! ToggleTranslateMode()
    if g:toggle_translate_mode == 0
        if g:toggle_grep_translate_mode == 1
            call ToggleGrepTranslateMode()
        endif
        let g:toggle_translate_mode = 1
        "execute "10new"
        execute "10split " . "TranslationBox.txt"
        let g:dump_buffer_nr = bufnr('%')
        setlocal noswapfile
        setlocal bufhidden=delete
        iabclear <buffer>
        "move back to working buffer
        execute "wincmd p"
        "nnoremap f :silent GrepCurrentWord<Cr>
        "translate word under cursor in normal mode: (f or \t both do the same):
        nnoremap f :silent TranslateCurrentWord<Cr>
        nnoremap <Leader>t :silent TranslateCurrentWord<Cr>
        vnoremap f "1y:silent call TranslateCurrentWord(@1)<Cr>
        vnoremap <Leader>t "1y:silent call TranslateCurrentWord(@1)<Cr>
        "do not use quotes, type the word without quotes and hit <Enter>...
        nnoremap <Leader>tr :TranslateFromCommandLine 
        "this key combo will take you to greptranslate in command line, already
        "with the rather sensible paramets -A1 -B1 -w -i:
        "nnoremap <Leader>gt :GrepTranslate -A1 -B1 -w -i 
        nnoremap <Leader>gt :GrepTranslate -w -i 
    else
        let g:toggle_translate_mode = 0
        execute "bdelete! " . g:dump_buffer_nr
        nunmap f
        vunmap f
        nunmap <Leader>tr
        nunmap <Leader>gt
        nunmap <Leader>t
        vunmap <Leader>t
    endif

endfunction

command ToggleTranslateMode call ToggleTranslateMode()

"the same stuff this time using grep and the dict file
"actually it is redundunt the translation functions are what's matter
let g:toggle_grep_translate_mode = 0

"I disable the GrepTranslateMode as it seems there is no need for it but I leave
"the code because it is simple and might be useful in debuging and further
"development
"function! ToggleGrepTranslateMode()
"    if g:toggle_grep_translate_mode == 0
"        if g:toggle_translate_mode == 1
"            call ToggleTranslateMode()
"        endif
"        "let g:toggle_translate_mode = 1
"        let g:toggle_grep_translate_mode = 1
"        execute "10split " . g:tempfile
"        let g:dump_buffer_nr = bufnr('%')
"        setlocal noswapfile
"        setlocal bufhidden=delete
"        iabclear <buffer>
"        "move back to working buffer
"        execute "wincmd p"
"        nnoremap f :silent GrepCurrentWord<Cr>
"    else
"        let g:toggle_translate_mode = 0
"        execute "bdelete! " . g:dump_buffer_nr
"        nunmap f
"    endif
"endfunction

"command ToggleGrepTranslateMode call ToggleGrepTranslateMode()

command -nargs=1 TranslateFromCommandLine call TranslateCurrentWord(<f-args>)

"now a command that allows passing on search parameters to grep as well as
"search term
function! GrepTranslate(input_string)
    if g:toggle_translate_mode == 0
        echo "error, translate mode is disabled"
        return 0
    else
        let this_buffer = bufnr('%')
        let temp_buffer = g:dump_buffer_nr
        let temp_bufwinnr = bufwinnr(temp_buffer)
        let this_bufwinnr = bufwinnr(this_buffer)
        if mode() == 'n'
            execute "normal"."\"1yw"
        endif
        if a:input_string == ''
            let searchword = @1
        else    
            let searchword = a:input_string
        endif
        "I consider adding a second argument for dict selection but for now...
        let my_dict = g:dict_file
        "execute "b".temp_buffer
        execute temp_bufwinnr."wincmd w" 
        execute "%d"
        execute "goto 1"
        "execute "silent read !dict -d fd-deu-eng ".searchword
        "I replaced dict with grep below:
        "let myline = "silent 0read !grep -w -i -h -m4 -A1 '^".searchword."' "
        let myline = "silent 0read !grep ".searchword." "
        "let myline .= g:plugin_dir.'/deu-eng.dict'
        "let myline .= g:dict_file
        let myline .= my_dict
        execute myline
python3 << EOF
import vim
my_temp_str = vim.eval("searchword")
my_temp_list = my_temp_str.split(' ')
my_word = my_temp_list[-1]
if (my_word[0] == '^'):
    my_word = my_word[1:]
vim.command("match Error /" + my_word + "\c/")
EOF
        "execute "b".this_buffer
        execute this_bufwinnr."wincmd w" 
        "execute "silent !dict -d fd-deu-eng <cword> > ~/tempfile.txt" 
    endif
endfunction

" for this command there is one argument typed as command line argument for
" grep, do not wrap it in quotes!
" which is the parameters and search word for grep. i.e: -A1 -B1 -w -i Apfel
command -nargs=1 GrepTranslate call GrepTranslate(<f-args>)

function! SwitchLanguage(dict_file, ...)
    let my_dict = a:dict_file
    if 1 <= a:0
        let my_path = a:1
    else
        let my_path = g:plugin_dir
    endif
    let g:dict_file = my_path.'/'.my_dict
endfunction

"enter the path as quoted string, the dirname mustnot end with a '\'. The
"dict_name is the filename, without the path...
