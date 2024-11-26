{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE UnicodeSyntax #-}
{-# LANGUAGE ImportQualifiedPost #-}
{-# OPTIONS_HADDOCK show-extensions #-}

{- | Justify Text app/module.

      reads text from stdin.
      takes 1 optional argument - desired linewidth.
      if not provided it defaults to 68.

      If the paragraphs designator is indentaion instead of newline, this will
      change to newline (empty line between paragraphs).

      Each paragraphs is center-justified to the desired linewidth by adding spaces.
      The spaces are randomly distributed between the words of the line.
      In case of words longer than the desired linewidth, such words make
      their own line, and will not be broken or hyphenated.

      If the program is given additional arguments other than linewidth,
      the output will be righ-justified text to the specification of the
      first argument (wrapped text) with no extra space padding

      ==== __Examples__
      use from the linux command line:
      `cat mytext.txt | justifyText-exe 80 > output.txt`

      for direct call from vim on a selected range range:
      `:'<,'>!justifyLine-exe 72`

      also can set in vim equalprg or formatprg,
      e.g
      `:set equalprg=justifyLine-exe\ 58`
-}
module Main (
  main,
) where

import Control.Applicative
import Control.Monad
import Data.Char
import Data.List
import Data.Semigroup
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import Data.Tuple
import Lib
import System.Environment
import System.IO
import System.Random qualified as R
import System.Random.Stateful qualified as RS

getRequiredSpaces :: Int -> Int -> Int -> [Int]
getRequiredSpaces justWidth lineWidth numPlaces
  | justWidth <= lineWidth = [0 | i <- [1 .. numPlaces]]
  | numPlaces == 1 = [justWidth - lineWidth]
  | numPlaces < 1 = [0]
  | otherwise =
      let spacesToFill = justWidth - lineWidth
          a = spacesToFill `div` numPlaces
          b = spacesToFill `mod` numPlaces
       in [a + 1 | i <- [1 .. b]] ++ [a | i <- [1 .. (numPlaces - b)]]

split :: (a -> Bool) -> [a] -> [[a]]
split p [] = []
split p xs =
  let (pref, rest) = break p xs
      suf = dropWhile p rest
   in pref : (split p suf)

prepLine :: T.Text -> T.Text
prepLine line
  | " " `T.isPrefixOf` line = T.concat ["\n", (T.strip line)]
  | otherwise = T.strip line

prepText :: T.Text -> T.Text
prepText text =
  let mylines = T.lines text
      plines = map prepLine mylines
   in T.unlines plines

preProcessLines :: [T.Text] -> [T.Text]
preProcessLines ls = do
  l <- ls
  let p = T.isPrefixOf (T.pack " ")
  if p l
    then return (T.concat [(T.pack "\n"), (T.strip l)])
    else return (T.strip l)

paragraphs :: T.Text -> [T.Text]
paragraphs text =
  let text' = T.strip $ prepText text
      mylines = T.lines text'
      mypars = split (== "") mylines
      mypars' = map (T.unlines) mypars
   in -- mypars' = map (T.intercalate "\n") mypars
      mypars'

chip :: Int -> T.Text -> [T.Text] -> T.Text
chip _ w [] = w
chip n "" (x : xs) = chip n x xs
chip n w (x : xs)
  | T.length w + 1 + T.length x <= n = chip n (T.concat [w, " ", x]) xs
  | otherwise = w

-- | right justrify [textWords] to <=n line width
chipChop :: Int -> T.Text -> T.Text -> [T.Text] -> T.Text
chipChop _ w txt [] = T.concat [txt, "\n", w]
chipChop n "" txt (x : xs) = chipChop n x txt xs
chipChop n w txt (x : xs)
  | T.length w + 1 + T.length x > n =
      if T.null txt
        then chipChop n "" w (x : xs)
        else chipChop n "" (T.concat [txt, "\n", w]) (x : xs)
  | otherwise = chipChop n (T.concat [w, " ", x]) txt xs

justifyLine :: Int -> T.Text -> T.Text
justifyLine n "" = ""
justifyLine n ln =
  let ws = T.words ln
      numWords = length ws - 1
      lineWidth = (sum (map T.length ws)) + (length ws) - 1
      gapToAddList = getRequiredSpaces n lineWidth numWords
      gaps = [T.replicate i " " | i <- gapToAddList] ++ [""]
      newWords = zipWith T.append ws gaps
   in T.unwords newWords

-- | randomized version of justifyLine
justifyLineR :: RS.StdGen -> Int -> T.Text -> T.Text
justifyLineR n _ "" = ""
justifyLineR g n ln =
  let ws = T.words ln
      numWords = length ws - 1
      lineWidth = (sum (map T.length ws)) + (length ws) - 1
      -- gapToAddList = permute g $ getRequiredSpaces n lineWidth numWords
      gapToAddList = permute' g 1 $ getRequiredSpaces n lineWidth numWords
      gaps = [T.replicate i " " | i <- gapToAddList] ++ [""]
      newWords = zipWith T.append ws gaps
   in T.unwords newWords

{- | wraps text so that each line is at most linewidth wide.
      doesn't add padding space.e
-}
wrapText :: Int -> T.Text -> T.Text
wrapText _ "" = ""
wrapText n txt =
  let text = prepText txt
      ps = paragraphs text
      pss = map ((chipChop n "" "") . (T.words)) ps
   in -- in (T.unlines pss)
      T.append (T.intercalate "\n\n" pss) "\n"

justifyText :: Int -> T.Text -> T.Text
justifyText _ "" = ""
justifyText n txt =
  let text = prepText txt
      ps = paragraphs text
      pss = map ((chipChop n "" "") . (T.words)) ps
      ls = map (justifyLine n) (T.lines (T.unlines pss))
      rjtxt = (T.intercalate "\n\n" pss)
      cjtxt = T.unlines $ map (justifyLine n) (T.lines rjtxt)
   in cjtxt


{- | a variation which leaves the last line of a paragraph left justified.
- -}
--justifyText' :: Int -> T.Text -> T.Text
--justifyText' _ "" = ""
--justifyText' n txt =
--  let text = prepText txt
--      ps = paragraphs text
--      pss = map ((chipChop n "" "") . (T.words)) ps
--      lss = map T.lines pss
--      f s = if T.last s /= '\n' then (justifyLine n s)
--                                    else s
--      lsss = map (map f) lss
--      lssss = map T.unlines lsss
--      cjtxt = T.unlines lssss
--   in cjtxt


-- | randomized version of justifyText
justifyTextR :: RS.StdGen -> Int -> T.Text -> T.Text
justifyTextR _ _ "" = ""
justifyTextR g n txt =
  let text = prepText txt
      ps = paragraphs text
      pss = map ((chipChop n "" "") . (T.words)) ps
      rjtxt = (T.intercalate "\n\n" pss)
      cjtxt = T.unlines $ map (justifyLineR g n) (T.lines rjtxt)
      pars = paragraphs cjtxt
      pars2 = map fixPar pars
      output = T.append (T.intercalate "\n\n" pars2) "\n"
      --output = T.unlines pars2
   --in cjtxt
   in output

fixLastLine :: Int -> T.Text -> T.Text
fixLastLine n myline
  | n<=0 = T.unwords $ T.words myline
  | otherwise = myline

fixPar :: T.Text -> T.Text
fixPar par = 
  let ls = T.lines par
      n = length ls
      ns = [(n-1),(n-2)..0]
      lss = zipWith fixLastLine ns ls
    --in T.intercalate ("\n") lss
    in T.init $ T.unlines lss

generateRandomPerm :: RS.StdGen -> Int -> Int -> [Int]
generateRandomPerm g n m =
  let f = (`mod` n)
      l = nub $ map f (take m (R.randoms g) :: [Int])
      new_g = snd (R.uniform g :: (Bool, R.StdGen))
   in if length l == n
        then l
        else generateRandomPerm new_g n (2 * m)

permute :: RS.StdGen -> [a] -> [a]
permute g [] = []
permute g (x : []) = (x : [])
permute g xs =
  let n = length xs
      p = generateRandomPerm g n (2 * n)
   in [xs !! i | i <- p]

testRandom :: Int -> [Word]
testRandom n =
  let rollsM :: RS.StatefulGen g m => Int -> g -> m [Word]
      rollsM n = replicateM n . RS.uniformRM (1, 6)
      pureGen = R.mkStdGen 42
   in RS.runStateGen_ pureGen (rollsM n) :: [Word]

getOutputFile :: String -> IO Handle
getOutputFile path = do
  hand <- openFile path WriteMode
  isOK <- hIsWritable hand
  if and [isOK, path /= "-"]
    then return hand
    else return stdout

main :: IO ()
main = do
  args <- getArgs
  text <- TIO.getContents
  g <- R.newStdGen
  -- ; text <- TIO.hGetLine stdin
  -- ; text <- TIO.readFile "sample.txt"
  let n =
        if null args
          then 68 :: Int
          else read (head args) :: Int
  let jtext = justifyTextR g n text
  let output =
        if length args <= 1
          then jtext
          else wrapText n text
  -- TIO.hPutStr stdout jtext
  TIO.hPutStr stdout output

testmain :: IO ()
testmain = do
  args <- getArgs
  -- ; text <- TIO.getContents
  text <- TIO.readFile "sample.txt"
  g <- R.newStdGen
  -- ; text <- TIO.hGetLine stdin
  let n =
        if null args
          then 68 :: Int
          else read (head args) :: Int
  let jtext = justifyTextR g n text
  -- ; let jtext = justifyText n text
  -- ; TIO.putStrLn "hello!"
  -- ; TIO.hPutStr stdout "good bye!"
  let output =
        if length args > 1
          then jtext
          else (wrapText n jtext)
  TIO.hPutStr stdout output

-- TIO.hPutStr stdout jtext

{- | 1-indexed swap operation on lists
-
-}
swapTwo :: Int -> Int -> [a] -> [a]
swapTwo i j xs
  | i < 1 = xs
  | j > length xs = xs
  | i == j = xs
  | i > j = swapTwo j i xs
  | otherwise =
      let prefix = take (i - 1) xs
          (li : taili) = drop (i - 1) xs
          midfix = take (j - i - 1) taili
          (lj : suffix) = drop (j - 1) xs
       in prefix ++ [lj] ++ midfix ++ [li] ++ suffix

permute' :: RS.StdGen -> Int -> [a] -> [a]
permute' _ _ [] = []
permute' g i (x : []) = x : []
permute' g i xs
  | i <= 0 = xs
  | i >= length xs = xs
  | otherwise =
      let n = length xs
          (j, g') = R.randomR (i, n) g
          ys = swapTwo i j xs
       in permute' g' (i + 1) ys





--textt = do TIO.readFile "pg22367.txt"
--sampleio = do TIO.readFile "sample.txt"
