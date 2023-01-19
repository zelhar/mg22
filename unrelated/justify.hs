{-# LANGUAGE OverloadedStrings #-}
import Control.Monad
import Control.Applicative
import System.IO
import Data.Char
import System.Environment
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Data.Semigroup
--import Data.String

--main :: IO ()
--mmain = do {
--          x <- getLine;
--          print x;
--          putStrLn x;
--          print (map toUpper x);
--          print $ x ++ "fff";
--          let y = map toUpper x
--          putStrLn ("hello " ++ y);
--          }
--
--

-- The plan:
-- strip white space
-- split lines
-- take paragraph, justify, repeat

--justifyParagraph :: Int -> [T.Text] -> T.Text


getOutputFile :: String -> IO Handle
getOutputFile path = do { hand <- openFile path WriteMode
                        ; isOK <- hIsWritable hand
                        ; if and [isOK, path /= "-"]  then return hand else return stdout
                        }

main = 
  do { args <- getArgs
     ; text <- TIO.getContents
     ; let minitext = T.takeWhile (/='ü') text
     ; TIO.putStrLn $ T.strip $ T.toUpper minitext
     ; let myLines = map T.strip (T.lines text)
     ; TIO.putStrLn $ T.unlines myLines
     }

mainOld =
 do { args <- getArgs
    ; let path = if null args then "-" else head args
    ; hand <- getOutputFile path
    --; hand <- openFile (head args) AppendMode
    --; hPutStrLn hand "fuck off!"
    ; print hand
    ; do { test <- hIsWritable hand
         ; print "is writeable?"
         ; print test
         }
    ; do { test <- hIsReadable hand
         ; print "is readable?"
         ; print test
         }
    ; print "is handle stdout?"
    ; print (stdout == hand)
    ; hPutStrLn stdout (show args)
    ; name <- getLine
    ; let loudName = map toUpper name
    ; putStrLn ("Hello " ++ loudName ++ "!")
    ; putStrLn ("Oh boy! Am I excited to meet you, " ++ loudName)
    ;hClose hand
    }

--mainss = do { name <- getLine
--            ; let loudName = map toUpper name
--            ; putStrLn ("Hello " ++ loudName ++ "!");
--            }
            
