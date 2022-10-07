{-# LANGUAGE OverloadedStrings #-}

module Main where

import Termonad.App (defaultMain)
import Termonad.Config
  ( FontConfig
  , FontSize(FontSizePoints)
  , Option(Set)
  , ShowScrollbar(ShowScrollbarAlways)
  , ShowScrollbar(ShowScrollbarIfNeeded)
  , ShowTabBar(ShowTabBarIfNeeded)
  , CursorBlinkMode(CursorBlinkModeOff)
  , defaultConfigOptions
  , defaultFontConfig
  , defaultTMConfig
  , fontConfig
  , fontFamily
  , fontSize
  , options
  , showScrollbar
  , showTabBar
  , cursorBlinkMode
  , showMenu
  )
import Termonad.Config.Colour
  ( AlphaColour, ColourConfig, addColourExtension, createColour
  , createColourExtension, cursorBgColour, defaultColourConfig
  )

-- | This sets the color of the cursor in the terminal.
--
-- This uses the "Data.Colour" module to define a dark-red color.
-- There are many default colors defined in "Data.Colour.Names".
cursBgColour :: AlphaColour Double
cursBgColour = createColour 204 0 0

-- | This sets the colors used for the terminal.  We only specify the background
-- color of the cursor.
colConf :: ColourConfig (AlphaColour Double)
colConf =
  defaultColourConfig
    { cursorBgColour = Set cursBgColour
    }

-- | This defines the font for the terminal.
fontConf :: FontConfig
fontConf =
  defaultFontConfig
    { --fontFamily = "DejaVu Sans Mono"
    fontFamily = "Monospace"
    , fontSize = FontSizePoints 13
    }

main :: IO ()
main = do
  colExt <- createColourExtension colConf
  let termonadConf =
        defaultTMConfig
          { options =
              defaultConfigOptions
                { fontConfig = fontConf
                  -- Make sure the scrollbar is always visible.
                --, showScrollbar = ShowScrollbarAlways
                , showScrollbar = ShowScrollbarIfNeeded
                , showTabBar = ShowTabBarIfNeeded
                , cursorBlinkMode = CursorBlinkModeOff
                , showMenu = False
                }
          }
        `addColourExtension` colExt
  defaultMain termonadConf
