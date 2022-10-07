module Main where
import XMonad
import qualified Data.Map as M
import           Data.Map                    (Map)
import           Data.Monoid                 (appEndo)
import           XMonad.Actions.CycleWS      (nextWS, prevWS, shiftToNext,
                                              shiftToPrev)
import           XMonad.Hooks.ManageDocks
import           XMonad.Hooks.EwmhDesktops
import           XMonad.Hooks.DynamicLog
import XMonad.Hooks.StatusBar
import XMonad.Hooks.StatusBar.PP
import XMonad.Util.Loggers
import XMonad.Util.Ungrab
import           XMonad.Hooks.ManageHelpers
import           XMonad.Layout.Circle        (Circle (..))
import           XMonad.Layout.NoBorders     (smartBorders)
import           XMonad.Layout.PerWorkspace  (onWorkspace)
import           XMonad.Layout.SimplestFloat (simplestFloat)
import           XMonad.Prompt.Shell         (shellPrompt)
import           XMonad.StackSet             (RationalRect (..), currentTag)
import Graphics.X11.ExtraTypes.XF86
import System.Exit
import XMonad.Actions.CycleWS
import XMonad.Actions.Promote
import XMonad.Actions.DwmPromote
import XMonad.Config.Xfce
import XMonad.Hooks.SetWMName
import XMonad.Layout
import XMonad.Layout.Accordion
import qualified XMonad.Layout.BinarySpacePartition as BSP
import XMonad.Layout.Column
import XMonad.Layout.Combo
import XMonad.Layout.ComboP
import XMonad.Layout.Cross
import XMonad.Layout.Grid
import XMonad.Layout.NoBorders
import XMonad.Layout.ResizableTile
import XMonad.Layout.Simplest
import XMonad.Layout.Spiral
import XMonad.Layout.TabBarDecoration
import XMonad.Layout.Tabbed
import XMonad.Layout.TwoPane
import XMonad.Layout.Reflect
import XMonad.Layout.TwoPanePersistent
import XMonad.Layout.WindowNavigation
import XMonad.Actions.WindowNavigation
import XMonad.Util.EZConfig
import XMonad.Util.Paste
import XMonad.Util.Run
import XMonad.Util.Themes
import XMonad.Util.SpawnOnce
import qualified XMonad.StackSet as W
import XMonad.Actions.WindowMenu
import XMonad.Actions.GridSelect
import XMonad.Config.Desktop





--main :: IO ()
--main = xmonad xfceConfig
--main = xmonad =<< xmobar def
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar (ewmh def)
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar (ewmh xfceConfig)
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar xfceConfig
--main = xmonad =<< 
--  withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<<
--    statusBar myBar myPP toggleStrutsKey xfceConfig
--
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar (ewmh def)
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar (ewmh xfceConfig)
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar (ewmh xfceConfig)
--  { terminal = "alacritty"
--  , modMask = mod4Mask
--  , keys = \c -> myKeys c `M.union` keys def c
--  , focusFollowsMouse = False
--  --, layoutHook = myLayout
--  , manageHook = manageHook xfceConfig <+> manageDocks
--  --, handleEventHook = handleEventHook defaultConfig <+> docksEventHook
--  , layoutHook = avoidStruts $ myLayout
--  , logHook = dynamicLog
--  --, layoutHook = avoidStruts $ layoutHook def
--  }

--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< xmobar myConfig
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d) =<< statusBar myBar myPP toggleStrutsKey myConfig
--  where
--    myConfig = docks xfceConfig
--      { terminal = "alacritty"
--      , modMask = mod4Mask
--      , keys = \c -> myKeys c `M.union` keys def c
--      , focusFollowsMouse = False
--      --, layoutHook = myLayout
--      , manageHook = manageHook xfceConfig <+> manageDocks
--      --, manageHook = manageDocks <+> manageHook def
--      --, handleEventHook = handleEventHook defaultConfig <+> docksEventHook
--      , layoutHook = avoidStruts $ myLayout
--      --, logHook = dynamicLog
--      --, logHook = do
--      --    dynamicLogWithPP xmobarPP
--      --    logHook desktopConfig
--      ----, layoutHook = avoidStruts $ layoutHook def
--      }
--
--myBar = "xmobar"
--myPP = xmobarPP
--toggleStrutsKey XConfig {XMonad.modMask = modMask} = (modMask, xK_b)
--main = xmonad def
-- Command to launch the bar.
--myBar = "xmobar"
---- Custom PP, configure it as you like. It determines what is being written to the bar.
--myPP = xmobarPP { ppCurrent = xmobarColor "#429942" "" . wrap "<" ">" }
---- Key binding to toggle the gap for the bar.
--toggleStrutsKey XConfig {XMonad.modMask = modMask} = (modMask, xK_b)


main :: IO ()
--main = xmonad =<< withWindowNavigation (xK_w, xK_a, xK_s, xK_d)
--      ( ewmhFullscreen 
--      . ewmh 
--      . xmobarProp 
--      . withEasySB (statusBarProp "xmobar" (pure myXmobarPP)) defToggleStrutsKey
--      $ myConfig)
main = xmonad
      . ewmhFullscreen 
      . ewmh 
      . xmobarProp 
      . withEasySB (statusBarProp "xmobar" (pure myXmobarPP)) defToggleStrutsKey
      $ myConfig


myConfig = def
    { modMask = mod4Mask
    , terminal = "alacritty"
    , keys = \c -> myKeys c `M.union` keys def c
    , focusFollowsMouse = False
    , layoutHook = myLayout
    , manageHook = myManageHook
    , startupHook        = myStartupHook
    }

myManageHook :: ManageHook
myManageHook = composeAll
    [ className =? "Gimp" --> doFloat
    , isDialog            --> doFloat
    ]


myModMask = mod4Mask

-- A list of custom keys
myKeys :: XConfig Layout -> Map (ButtonMask, KeySym) (X ())
myKeys (XConfig {modMask = myModMask}) = M.fromList $
    [ -- Some programs
      ((myModMask, xK_p), spawn "dmenu_run")
    , ((myModMask .|. shiftMask, xK_p), spawn "dmenu_run")
    --, ((myModMask, xK_a), sendMessage ToggleStruts)
    , ((myModMask, xK_z), sendMessage ToggleStruts)
      -- Full float
    , ((myModMask, xK_f), fullFloatFocused)
    , ((myModMask, xK_t), withFocused $ windows . W.sink)
    , ((noModMask, xK_Menu),  windowMenu)
    --screen swapping
    , ((myModMask, xK_g), goToSelected def)
    , ((myModMask .|. shiftMask, xK_o), shiftNextScreen)
    , ((myModMask .|. shiftMask, xK_f), moveTo Next emptyWS)
    , ((myModMask .|. shiftMask,               xK_z),     toggleWS)
    --combo mode layout navigation
    , ((myModMask, xK_Left), sendMessage $ Go L)
    , ((myModMask, xK_Right), sendMessage $ Go R)
    , ((myModMask, xK_Up), sendMessage $ Go U)
    , ((myModMask, xK_Down), sendMessage $ Go D)
    , ((myModMask .|. controlMask, xK_Left), sendMessage $ Move L)
    , ((myModMask .|. controlMask, xK_Right), sendMessage $ Move R)
    , ((myModMask .|. controlMask, xK_Up), sendMessage $ Move U)
    , ((myModMask .|. controlMask, xK_Down), sendMessage $ Move D)
    , ((myModMask .|. controlMask, xK_s), sendMessage $ SwapWindow)
    , ((myModMask .|. shiftMask, xK_Left), sendMessage $ Swap L)
    , ((myModMask .|. shiftMask, xK_Right), sendMessage $ Swap R)
    , ((myModMask .|. shiftMask, xK_Up), sendMessage $ Swap U)
    , ((myModMask .|. shiftMask, xK_Down), sendMessage $ Swap D)
    ]
  where
    -- Function to fullFloat a window
    fullFloatFocused = withFocused $ \f -> windows =<< appEndo `fmap` runQuery
                          doFullFloat f

    -- Function to rectFloat a window
    rectFloatFocused = withFocused $ \f -> windows =<< appEndo `fmap` runQuery
                          (doRectFloat $ RationalRect 0.02 0.05 0.96 0.9) f


myLayout =windowNavigation $ smartBorders . avoidStruts $ 
--myLayout =windowNavigation $ smartBorders $ 
    Full
    ||| BSP.emptyBSP
    ||| (reflectHoriz $ combineTwo (TwoPane 0.01 0.5) (tabbed shrinkText tabConfig) (tabbed shrinkText tabConfig))
    ||| combineTwo (TwoPane 0.01 0.5) (Simplest) (Simplest)
    ||| tabbed shrinkText tabConfig
--    ||| Mirror $ (combineTwo (TwoPane 0.01 0.5) (Simplest) (Simplest))
--    ||| combineTwo (TwoPane 0.03 0.5) (tabbed shrinkText tabConfig) (tabbed shrinkText tabConfig)
--    ||| windowNavigation (Mirror emptyBSP)
--    ||| (layoutHook defaultConfig)
--    ||| Tall 1 (1/100) (1/2)
---    ||| emptyBSP
--    ||| Column 1.6
--    ||| simpleCross
--    ||| Circle
    -- ||| TwoPanePersistent Nothing (1/100) (1/2)
--    ||| spiral(6/7)
--    ||| Simplest
--    ||| windowNavigation (Mirror $ combineTwo (TwoPane 0.03 0.5) (tabbed shrinkText tabConfig) (tabbed shrinkText tabConfig))
--    ||| Accordion
  where
    tabConfig = theme smallClean
    --tabConfig = defaultTheme


myXmobarPP :: PP
myXmobarPP = def
    { ppSep             = magenta " • "
    , ppTitleSanitize   = xmobarStrip
    , ppCurrent         = wrap " " "" . xmobarBorder "Top" "#8be9fd" 2
    , ppHidden          = white . wrap " " ""
    , ppHiddenNoWindows = lowWhite . wrap " " ""
    , ppUrgent          = red . wrap (yellow "!") (yellow "!")
    , ppOrder           = \[ws, l, _, wins] -> [ws, l, wins]
    , ppExtras          = [logTitles formatFocused formatUnfocused]
    }
  where
    formatFocused   = wrap (white    "[") (white    "]") . magenta . ppWindow
    formatUnfocused = wrap (lowWhite "[") (lowWhite "]") . blue    . ppWindow

    -- | Windows should have *some* title, which should not not exceed a
    -- sane length.
    ppWindow :: String -> String
    ppWindow = xmobarRaw . (\w -> if null w then "untitled" else w) . shorten 30

    blue, lowWhite, magenta, red, white, yellow :: String -> String
    magenta  = xmobarColor "#ff79c6" ""
    blue     = xmobarColor "#bd93f9" ""
    white    = xmobarColor "#f8f8f2" ""
    yellow   = xmobarColor "#f1fa8c" ""
    red      = xmobarColor "#ff5555" ""
    lowWhite = xmobarColor "#bbbbbb" ""

myStartupHook = do
  spawnOnce "~/bin/myxkb.sh"
