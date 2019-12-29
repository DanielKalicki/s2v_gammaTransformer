call plug#begin()
Plug 'scrooloose/nerdtree'
	"Plug 'Xuyuanp/nerdtree-git-plugin' not working
	"Plug 'tiagofumo/vim-nerdtree-syntax-highlight' not working
Plug 'morhetz/gruvbox'
Plug 'dracula/vim', { 'name': 'dracula' }
Plug 'jistr/vim-nerdtree-tabs'
Plug 'itchyny/lightline.vim'
Plug 'scrooloose/nerdcommenter'
Plug 'blueyed/vim-diminactive'
Plug 'tpope/vim-fugitive'
Plug 'ycm-core/YouCompleteMe'
	"Plug 'ajmwagar/vim-deus'
	"Plug 'NLKNguyen/papercolor-theme'
	"Plug 'mkarmona/materialbox'
Plug 'sheerun/vim-polyglot'
	"Plug 'vim-airline/vim-airline'
	"Plug 'vim-airline/vim-airline-themes'
Plug 'airblade/vim-gitgutter'
Plug 'tmhedberg/SimpylFold'
	"Plug 'vim-scripts/vim-auto-save'
Plug 'vim-syntastic/syntastic'	" pip3 install pylint flake8 pyflakes --user
	"Plug 'nvie/vim-flake8'	" apt-get install flake8
Plug 'luochen1990/rainbow'
Plug 'ryanoasis/vim-devicons'
	"Plug 'w0rp/ale'
	"Plug 'maximbaz/lightline-ale'
Plug 'Yggdroot/indentLine'
call plug#end()

set encoding=UTF-8

set number
set relativenumber
set incsearch

set wildmenu
set wildmode=list:longest,full

set splitbelow splitright

set termguicolors 
"gruvbox colorscheme

set background=dark

let gruvbox_contrast_dark='soft'
colorscheme gruvbox

	"colorscheme deus

	"set background=light
	"colorscheme PaperColor

	"colorscheme materialbox

let g:nerdtree_tabs_open_on_console_startup=1
let g:NERDTreeGitStatusWithFlags = 1

let g:lightline = {
      \ 'active': {
      \   'left': [ [ 'mode', 'paste' ],
      \             [ 'gitbranch', 'readonly', 'filename', 'modified' ] ]
      \ },
      \ 'component_function': {
      \   'gitbranch': 'fugitive#head'
      \ },
      \ }

	" let g:lightline.component_expand = {
	"       \  'linter_checking': 'lightline#ale#checking',
	"       \  'linter_warnings': 'lightline#ale#warnings',
	"       \  'linter_errors': 'lightline#ale#errors',
	"       \  'linter_ok': 'lightline#ale#ok',
	"       \ }
	" let g:lightline.component_type = {
	"       \     'linter_checking': 'left',
	"       \     'linter_warnings': 'warning',
	"       \     'linter_errors': 'error',
	"       \     'linter_ok': 'left',
	"       \ }
	" let g:lightline.active = { 'right': [[ 'linter_checking', 'linter_errors', 'linter_warnings', 'linter_ok' ]] }
set noshowmode " dont show the mode (-- INSERT --) in the last line

nnoremap <C-e> 5<C-e>
nnoremap <C-y> 5<C-y>

map <C-h> <C-w>h
map <C-j> <C-w>j
map <C-k> <C-w>k
map <C-l> <C-w>l

set tabstop=4
set expandtab       " Expand TABs to spaces

set mouse=a

let g:python_highlight_all = 1

	"let g:airline#extensions#tabline#enabled = 1

set updatetime=100
let g:gitgutter_max_signs = 5000

" Enable folding
	set foldmethod=indent
	set foldlevel=99
	nnoremap <space> za " Enable folding with the spacebar
	let g:SimpylFold_docstring_preview = 1

let g:auto_save = 1  " enable AutoSave on Vim startup

"for Plug syntastic
	execute pathogen#infect()
	set statusline+=%#warningmsg#
	set statusline+=%{SyntasticStatuslineFlag()}
	set statusline+=%*
	let g:syntastic_always_populate_loc_list = 1
	let g:syntastic_auto_loc_list = 0
	let g:syntastic_check_on_open = 1
	let g:syntastic_check_on_wq = 0
	let g:syntastic_python_python_exec = '/usr/bin/python3'
	augroup syntastic
	    autocmd CursorHold * nested update
	augroup END

set clipboard=unnamed

"autocmd BufWritePost *.py call Flake8()

let g:rainbow_active = 1

" j/k will move virtual lines (lines that wrap)
noremap <silent> <expr> j (v:count == 0 ? 'gj' : 'j')
noremap <silent> <expr> k (v:count == 0 ? 'gk' : 'k')

set bs=2 "Backspace with this value allows to use the backspace character (aka CTRL-H or "<-") to use for moving the cursor over automatically inserted indentation and over the start/end of line. (see also the whichwrap option) 

set colorcolumn=80
"highlight ColorColumn guibg=#800020

"set listchars+=eol:$,nbsp:_,tab:>-,trail:~,extends:>,precedes:<

"set list
"set listchars+=space:␣

highlight LineNr guifg=#b3754d

set scrolloff=5

" Highlight the cursor after a jump
" https://vim.fandom.com/wiki/Highlight_cursor_line_after_cursor_jump
function s:Cursor_Moved()
  let cur_pos = winline()
  if g:last_pos == 0
    set cul
    let g:last_pos = cur_pos
    return
  endif
  let diff = g:last_pos - cur_pos
  if diff > 1 || diff < -1
    set cul
  else
    set nocul
  endif
  let g:last_pos = cur_pos
endfunction
autocmd CursorMoved,CursorMovedI * call s:Cursor_Moved()
let g:last_pos = 0

" Key cheetsheet
" 0   move to beginning of line
" $   move to end of line
" _   move to first non-blank character of the line
" g_  move to last non-blank character of the line
" ngg move to n'th line of file (n is a number; 12gg moves to line 12)
" nG  move to n'th line of file (n is a number; 12G moves to line 12)
" %   jump to matching bracket { } [ ] ( )
" fX  to next 'X' after cursor, in the same line (X is any character)
" FX  to previous 'X' before cursor (f and F put the cursor on X)
" ;   repeat above, in same direction
" ,   repeat above, in reverse direction


function! HighlightWordUnderCursor()
    if getline(".")[col(".")-1] !~# '[[:punct:][:blank:]]' 
        exec 'match' 'StatusLine' '/\V\<'.expand('<cword>').'\>/' 
    else 
        match none 
    endif
endfunction

autocmd! CursorHold,CursorHoldI * call HighlightWordUnderCursor()

au FocusGained,BufEnter * :silent! !

" https://vim.fandom.com/wiki/Insert_a_single_character
" this is delete and replace with one character
nnoremap S :exec "normal s".nr2char(getchar())."\e"<CR>

" highlight Cursor guifg=black guibg=white
" set guicursor=n-v-c:block-Cursor

highlight LineNr guibg=#3c3836

" easier moving of code blocks
vnoremap <A-h> <gv
vnoremap <A-l> >gv

nnoremap <A-j> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
inoremap <A-j> <Esc>:m .+1<CR>==gi
inoremap <A-k> <Esc>:m .-2<CR>==gi
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv
