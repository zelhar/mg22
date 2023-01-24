{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE UnicodeSyntax     #-}

module Main ( main) where

import Control.Applicative
import Control.Monad
import Data.Char
import Data.List
import Data.Semigroup
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Lib
import System.Environment
import System.IO
import qualified System.Random as R
import qualified System.Random.Stateful as RS
import Data.Tuple

--main :: IO ()
--main = someFunc

getRequiredSpaces ∷ Int → Int → Int → [Int]
getRequiredSpaces justWidth lineWidth numWords
  | justWidth <= lineWidth = [0 | i <- [1..numWords]]
  | numWords <= 1 = [0]
  | otherwise = let numGaps = justWidth - lineWidth
                    a = numGaps `div` numWords
                    b = numGaps `mod` numWords
                 in [a+1 | i<-[1..b]] ++ [a | i<-[1..(numWords - b)]]


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



preProcessLines ∷ [T.Text] → [T.Text]
preProcessLines ls =
  do l <- ls
     let p = T.isPrefixOf (T.pack " ")
     if p l then return (T.concat [(T.pack "\n"), (T.strip l)])
            else return (T.strip l)


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


-- right justrify text to <=n line width
chipChop :: Int -> T.Text -> T.Text -> [T.Text] -> T.Text
chipChop _ w txt [] = T.concat [txt, "\n", w]
chipChop n "" txt (x:xs) = chipChop n x txt xs
chipChop n w txt (x:xs)
  | T.length w + 1 + T.length x > n = chipChop n "" (T.concat [txt, "\n", w]) (x:xs)
  | otherwise = chipChop n (T.concat [w, " ", x]) txt xs


justifyLine :: Int -> T.Text -> T.Text
justifyLine n "" = ""
justifyLine n ln =
  let ws = T.words ln
      numWords = length ws - 1
      lineWidth = (sum (map T.length ws)) + (length ws) - 1
      gapToAddList = getRequiredSpaces n lineWidth numWords
      gaps = [ T.replicate i " " | i <- gapToAddList ] ++ [""]
      newWords = zipWith T.append ws gaps
   in T.unwords newWords

wrapText :: Int -> T.Text -> T.Text
wrapText _ "" = ""
wrapText n txt =
  let text = prepText txt
      ps = paragraphs text
      pss = map ((chipChop n "" "") . (T.words)) ps
   in (T.unlines pss)
      
justifyText :: Int -> T.Text -> T.Text
justifyText _ "" = ""
justifyText n txt = 
  let text = prepText txt
      ps = paragraphs text
      pss = map ((chipChop n "" "").(T.words)) ps
      ls = map (justifyLine n) (T.lines (T.unlines pss))
      rjtxt = (T.unlines pss)
      cjtxt = T.unlines $ map (justifyLine n) (T.lines rjtxt)
      --pss = do { p <- ps
      --         ; let pp = chipChop n "" "" (T.words p)
      --         ; let ls = map (justifyLine n) (T.lines pp)
      --         ; [T.unlines ls]
      --         }
  --in (T.unlines ls)
  --in (T.unlines pss)
   in cjtxt
      

--generateRandomPerm' :: (R.RandomGen g) => g -> Int -> Int -> [Int]
--generateRandomPerm' :: RS.StdGen -> Int -> Int -> [Int]
--generateRandomPerm' g n m =
--  let f = (`mod` n)
--      l::[Int] = nub $ map f (take m (R.randoms g))
--      new_g = snd (R.uniform g :: (Bool, R.StdGen))
--   in if length l == n then l 
--                       else generateRandomPerm' new_g n (2*m) 

--generateRandomPerm :: Int -> Int -> [Int]
--generateRandomPerm n m =
--  let gen = R.mkStdGen (42*m)
--      f = (`mod` n)
--      l::[Int] = nub $ map f (take m (R.randoms gen))
--   in if length l == n then l
--                       else generateRandomPerm n (2*m)
      

testRandom :: Int -> [Word]
testRandom n = 
  let rollsM :: RS.StatefulGen g m => Int -> g -> m [Word]
      rollsM n = replicateM n . RS.uniformRM (1,6)
      pureGen = R.mkStdGen 42
   in RS.runStateGen_ pureGen (rollsM n) :: [Word]



getOutputFile ∷ String → IO Handle
getOutputFile path = do { hand <- openFile path WriteMode
                        ; isOK <- hIsWritable hand
                        ; if and [isOK, path /= "-"]  then return hand else return stdout
                        }


main :: IO ()
main = do { sample <- TIO.readFile "sample.txt"
          ; TIO.putStrLn sample
          ; let text = prepText sample
          ; TIO.putStrLn text
          }

test :: IO T.Text
test = do { sample <- TIO.readFile "sample.txt"
          --; TIO.putStrLn sample
          ; let text = prepText sample
          --; TIO.putStrLn text
          ; let pars = paragraphs text
          ; TIO.putStrLn (pars !! 2)
          ; return $ pars !! 2
          }

test2 :: IO T.Text
test2 = do { sample <- TIO.readFile "sample.txt"
          --; TIO.putStrLn sample
          ; let text = prepText sample
          --; TIO.putStrLn text
          ; let pars = paragraphs text
          ; TIO.putStrLn (pars !! 2)
          ; let text2 =  pars !! 2
          ; let text3 = justifyText 48 text2
          ; TIO.putStrLn text3
          ; return text3
          }

test3 n = do
  sample <- TIO.readFile "sample.txt"
  let text = prepText sample
  let pars = paragraphs text
  let parpars = map (T.words) pars
  let wpars = map (chipChop n "" "") parpars
  --return wpars
  TIO.putStrLn (T.unlines wpars)
  let wtext = (T.unlines wpars)
  return (T.intercalate "\n" wpars)

test4 n = do
  sample <- TIO.readFile "sample.txt"
  let text = wrapText n sample
  TIO.putStrLn text
  let mylines = map (justifyLine n) (T.lines text)
  let jtext = T.unlines mylines
  TIO.putStrLn jtext
  return jtext



textt = do TIO.readFile "pg22367.txt"
sampleio = do TIO.readFile "sample.txt"

foo = do { txt <- sampleio
         ; let mytext = justifyText 48 txt
         ; return mytext
         }

--sample <- sampleio
--txt = prepText sample
--TIO.putStrLn sample
--TIO.putStrLn txt
