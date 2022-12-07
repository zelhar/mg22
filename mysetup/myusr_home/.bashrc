PS1="\u@\h:\w> "
#PATH=/bin:/usr/bin:/usr/sbin:/usr/local/package/bin:/usr/local/bin:$HOME/bin
export PS1
#export PATH
export PATH=$HOME/bin:$PATH
export PATH=$HOME/.local/bin:$PATH
export PATH=$HOME/.ghcup/bin:$PATH

umask 007

alias ipython="ipython --no-autoindent"
alias import="-0-0-0-0-0-0-0" #cancel this cpmmand which hangs the Xsystem
alias rscript="Rscript --no-init-file --slave"

[ -f ~/.fzf.bash ] && source ~/.fzf.bash
[ -f "/home/ykolb/.ghcup/env" ] && source "/home/ykolb/.ghcup/env" # ghcup-env

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ykolb/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ykolb/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/home/ykolb/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/home/ykolb/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/home/ykolb/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/home/ykolb/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

