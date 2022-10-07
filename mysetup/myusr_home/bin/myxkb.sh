# this is the settinggs of my xkeyboard
# see man xkeyboard-config, man setxkbmap
#"setxkbmap -option "caps:ctrl_modifier"
#setxkbmap -option "ctrl:swapcaps"
#setxkbmap -option "caps:swapescape"
setxkbmap -option "shift:both_capslock_cancel"
setxkbmap -option "ctrl:nocaps"
setxkbmap -model pc105 -layout us,de,il \
    -variant altgr-intl, \
    -option grp:alt_space_toggle, \
    -option terminate:ctrl_alt_bksp, \
    -option compose:rctrl-altgr, \
    #-option ctrl:nocaps, \
    #-option shift:both_capslock_cancel, \
xcape -e 'Control_L=Escape'
