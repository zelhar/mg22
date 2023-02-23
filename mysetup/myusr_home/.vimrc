set nocompatible
" Remove ALL autocommands for the current group:
autocmd!
"------- START Plug manager instead of Vundle
" Specify a directory for plugins (for Neovim: ~/.local/share/nvim/plugged)
call plug#begin('~/.local/share/vim/plugged')
"Plug 'dhruvasagar/vim-table-mode'
"Plug 'godlygeek/tabular'
"Plug 'vim-scripts/TextFormat'

"Color Themes
Plug 'robertmeta/nofrils'
Plug 'andreypopp/vim-colors-plain'
Plug 'pbrisbin/vim-colors-off'
Plug 'vietjtnguyen/toy-blocks'
Plug 'drewtempelmeyer/palenight.vim'
Plug 'NLKNguyen/papercolor-theme'
Plug 'mswift42/vim-themes' "emacs themes
Plug 'chriskempson/base16-vim' 
Plug 'rafi/awesome-vim-colorschemes'

"Plugins I am testing to see if they are worth using
"Plug 'bling/vim-bufferline'
"Plug 'junegunn/fzf'
"Plug 'mileszs/ack.vim'

"Plug 'vim-airline/vim-airline'
"Plug 'vim-airline/vim-airline-themes'
"Plug 'tpope/vim-surround'
"Plug 'vim-scripts/loremipsum'
"Plug 'echuraev/translate-shell.vim'
"Plug 'vim-pandoc/vim-pandoc-syntax'
"vim-slime
"Plug 'jpalardy/vim-slime'
"coc-nvim
"Plug 'neoclide/coc.nvim', {'branch': 'release'} "not just haskell.

"haskell plugins
"Plug 'Twinside/vim-hoogle' "haskell hoogle plgin
"Plug 'neovimhaskell/haskell-vim' "syntax highlighter

"snakemake
"Plug 'snakemake/snakefmt'
"Plug 'snakemake/snakemake', {'rtp': 'misc/vim'}

"" Initialize plugin system
call plug#end()
"------- END Plug manager instead of Vundle

"make sure it stays on even if I delete Vundle or Neobunlde et al.
filetype plugin indent on

"set termguicolors

"Switch on syntax highlighting if it wasn't on yet.
if !exists("syntax_on")
    syntax on
endif

"make backspace function like normal apps in insert mode
set backspace=indent,eol,start

set textwidth=68
set background=dark

set t_Co=256
if !(&term =~ "rxvt")
    set t_8f=[38;2;%lu;%lu;%lum        " set foreground color
    set t_8b=[48;2;%lu;%lu;%lum        " set background color
    set t_Co=256
    set termguicolors
else
    set t_Co=256
    set notermguicolors
endif

"bracket highlight (that's on by default):
"set matchpairs=(:),{:},[:]

" Function that highlights text passing textwidth. Toggle with '\l'
noremap <silent> <Leader>l
      \ :if exists('w:long_line_match') <Bar>
      \   silent! call matchdelete(w:long_line_match) <Bar>
      \   unlet w:long_line_match <Bar>
      \ elseif &textwidth > 0 <Bar>
      \   let w:long_line_match = matchadd('ErrorMsg', '\%>'.&tw.'v.\+', -1) <Bar>
      \ else <Bar>
      \   let w:long_line_match = matchadd('ErrorMsg', '\%>80v.\+', -1) <Bar>
      \ endif<CR>

set expandtab
set smarttab
set softtabstop=4
set tabstop=4
set nu!
set shiftwidth=4

set spelllang=en,de,es,he
set nospell

" Read changes to file made by other applications
set autoread

"turn on auto-smart-indent
set autoindent
set smartindent


" Tell vim which characters to show for expanded TABs,
" trailing whitespace, and end-of-lines. VERY useful!
scriptencoding utf-8
if has("gui_running")
    set listchars=eol:¶,tab:>»,trail:·
"    set listchars=eol:¬,tab:>»,trail:·
else
"    set listchars=eol:¶,tab:>»,trail:·
    set listchars=eol:¬,tab:>»,trail:·
endif

" Show whitespace
"set list

set laststatus=2
set showtabline=2

" key-combination to move to next/previous buffer and tab (<C-l> same as redraw!)
nnoremap <Leader>] :bn<CR>
nnoremap <Leader>[ :bp<CR>
nnoremap <Tab> :tabnext<Cr>
nnoremap <C-Tab> :tabnext<Cr>
nnoremap <S-Tab> :tabprevious<Cr>
tnoremap <C-Tab> <C-\><C-n>:tabnext<Cr>

"splits the line after cursor and remain in normal mode
nnoremap <Leader><Enter> o<Esc>
nnoremap <M-Enter> i<Enter><Esc>

set clipboard+=unnamed
set clipboard+=unnamedplus


"clear forgotten popup windows
nnoremap <Leader>p :call popup_clear()<Cr>

"make esc work as expected in neovim terminal:
tnoremap <Esc> <C-\><C-n>
tnoremap <A-h> <C-\><C-N><C-w>h
tnoremap <A-j> <C-\><C-N><C-w>j
tnoremap <A-k> <C-\><C-N><C-w>k
tnoremap <A-l> <C-\><C-N><C-w>l
inoremap <A-h> <C-\><C-N><C-w>h
inoremap <A-j> <C-\><C-N><C-w>j
inoremap <A-k> <C-\><C-N><C-w>k
inoremap <A-l> <C-\><C-N><C-w>l
nnoremap <A-h> <C-w>h
nnoremap <A-j> <C-w>j
nnoremap <A-k> <C-w>k
nnoremap <A-l> <C-w>l

"highlight all matches to search results
set hlsearch
" highlight match while still typing search pattern
set incsearch
" make default search ignore case
set ignorecase

set sessionoptions=blank,buffers,curdir,folds,slash,unix,tabpages

au BufNewFile,BufRead *.hs setlocal nospell

"Make vim save backup of the files: 
set backup
set backupcopy=auto
"And save in the first folder it can of the following:
set backupdir=/run/media/zelhar/yjk-16g-msd/backupvimtexts/,
            \/run/media/zelhar/yjk-B16gb/backupvimtexts,
            \/run/media/zelhar/UF16/backupvimtexts,
            \/run/media/zelhar/JetFlash16/backupvimtexts,~/tmp,~/temp,.,~/,
            \/media/JetFlash16
"add a dictionary file for word completion (i_CTRL-X_CTRL-K):
"let g:symbols_file = "$HOME/dictionaries/symbols"
set dictionary+=$HOME/dictionaries/symbols
set dictionary+=$HOME/dictionaries/chemical_formulas.txt
set dictionary+=/usr/share/dict/american-english
set dictionary+=/usr/share/dict/ngerman
set dictionary+=/usr/share/dict/spanish
"make autocomplete (:help cpt) with ctrl-n search in the also in the dictionary
"set complete+=k
"set complete+=i
"set complete+=t
"set complete+=kspell
set completeopt=menuone,preview,longest,noinsert

"Set (locally) working dir to be the same as the file being edited in the buffer
autocmd BufEnter * silent! lcd %:p:h
"redraw screen when switching buffer, and returning to window (cleans garbage)  
""autocmd BufEnter * :redraw!
""autocmd FocusGained * :redraw! 
autocmd WinEnter * :filetype detect
"autocmd BufEnter * :filetype detect

" (Brace face)
set showmatch

" Set utf8 as standard encoding and en_US as the standard language
set encoding=utf8

" Use Unix as the standard file type
set ffs=unix,dos,mac

" In many terminal emulators the mouse works just fine, thus enable it.
if has('mouse')
  set mouse=a
endif
set mousehide		" Hide the mouse when typing text

set ch=2		" Make command line two lines high cmdheight
set cursorline
"set cc=81

" Setting scrolloff so cursor alsways stays inside that range except the top/bot
set scrolloff=5
"set a shorter timeout for key-combs and commands (default=1000)
set timeoutlen=1200
"set timeoutlen=700
set showcmd
"set position for new split windows:
set splitbelow
set splitright

" Test section - temp changes

"Neovim stuff
"trying to fix arrow keys in insert mode in VIM
if !has('nvim')
    set t_ku=OA
    set t_kd=OB
    set t_kr=OC
    set t_kl=OD
    imap <Esc>[A <Up>
    imap <Esc>[B <Down>
    imap <Esc>[C <Right>
    imap <Esc>[D <Left>
endif


"setting colorscheme variables
colorscheme zelhar-darkblue

" Don't pass messages to |ins-completion-menu|.
set shortmess+=c
"set signcolumn=yes
set signcolumn=number
"set hidden
set nohidden

" Use K to show documentation in preview window.
nnoremap <silent> K :call <SID>show_documentation()<CR>

function! s:show_documentation()
  if (index(['vim','help'], &filetype) >= 0)
    execute 'h '.expand('<cword>')
  elseif (coc#rpc#ready())
    call CocActionAsync('doHover')
  else
    execute '!' . &keywordprg . " " . expand('<cword>')
  endif
endfunction

" Having longer updatetime (default is 4000 ms = 4 s) leads to noticeable
" delays and poor user experience.
set updatetime=300
"snakefmt
