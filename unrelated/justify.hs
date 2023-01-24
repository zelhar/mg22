{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE UnicodeSyntax     #-}
import           Control.Applicative
import           Control.Monad
import           Data.Char
import           Data.List
import           Data.Semigroup
import qualified Data.Text           as T
import qualified Data.Text.IO        as TIO
import           System.Environment
import           System.IO

-- The plan:
-- strip white space
-- split lines
-- take paragraph, justify, repeat

--justifyParagraph :: Int -> [T.Text] -> T.Text

split ∷ (a → Bool) → [a] → [[a]]
split p [] = []
split p xs =
  let (pref, rest) = break p xs
      suf = dropWhile p rest
  in pref : (split p suf)

prepLine ∷ T.Text → T.Text
prepLine line
  | " " `T.isPrefixOf` line = T.concat ["\n", (T.strip line)]
  | otherwise = T.strip line

prepText :: T.Text -> T.Text
prepText text =
  let mylines = T.lines text
      plines = map prepLine mylines
   in T.unlines plines

paragraphs ∷ T.Text → [T.Text]
paragraphs text =
  let text' = T.strip $ prepText text
      mylines = T.lines text'
      mypars = split (=="") mylines
      mypars' = map (T.unlines) mypars
      --mypars' = map (T.intercalate "\n") mypars
   in mypars'

chip :: Int -> T.Text -> [T.Text] -> T.Text
chip _ w [] = w
chip n "" (x:xs) = chip n x xs
chip n w (x:xs)
  | T.length w + 1 + T.length x <= n = chip n (T.concat [w, " ", x]) xs
  | otherwise = w

chipChop :: Int -> T.Text -> T.Text -> [T.Text] -> T.Text
chipChop _ w txt [] = txt
chipChop n "" txt (x:xs) = chipChop n x txt xs
chipChop n w txt (x:xs)
  | T.length w + 1 + T.length x > n = chipChop n "" (T.concat [txt, "\n", w]) xs
  | otherwise = chipChop n (T.concat [w, " ", x]) txt xs

--chop :: Int -> T.Text -> T.Text
--chop n text
--  | n <= 0 = text
--  | T.all (isSpace) text = ""
--  | otherwise =
--    let w:ws = T.words text


preProcessLines ∷ [T.Text] → [T.Text]
preProcessLines ls =
  do l <- ls
     let p = T.isPrefixOf (T.pack " ")
     if p l then return (T.concat [(T.pack "\n"), (T.strip l)])
            else return (T.strip l)

--paragraphs :: T.Text -> [T.Text]
--paragraphs [] = []
--paragraphs xs =
--  let ls = lines xs
--      p = (=="\n")
--      q = (isPrefixOf "  ")
--      s = \l -> (p l) || (q l)

getRequiredSpaces ∷ Int → Int → Int → [Int]
getRequiredSpaces justWidth lineWidth numWords
  | justWidth <= lineWidth = []
  | otherwise = let numGaps = justWidth - lineWidth
                    a = numGaps `div` numWords
                    b = numGaps `mod` numWords
                 in [a+1 | i<-[1..b]] ++ [a | i<-[1..(numWords - b)]]

extractParagraph ∷ [T.Text] → [T.Text]
extractParagraph [] = []
extractParagraph ls = takeWhile (/="") (map T.strip ls)

extractParagraphs ∷ T.Text → [[T.Text]]
extractParagraphs text =
  let ls = map T.strip (T.lines text)
      ws = map T.words ls
      wss = do { l <- ws
               ; [map T.strip l]
               }
   in wss

getOutputFile ∷ String → IO Handle
getOutputFile path = do { hand <- openFile path WriteMode
                        ; isOK <- hIsWritable hand
                        ; if and [isOK, path /= "-"]  then return hand else return stdout
                        }

textt = do TIO.readFile "pg22367.txt"

sampleio = do TIO.readFile "sample.txt"

--text <- textt
--ts = extractParagraphs text
--tss = map T.unwords ts
--tsss = T.unlines tss


--paragraphs :: T.Text -> [T.Text]
--paragraphs text = let ls = T.lines text
--                  in ...

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


