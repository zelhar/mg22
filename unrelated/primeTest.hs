import Data.List
import System.Environment
import Text.Read



isPrime :: (Integral a) => a -> Bool
isPrime x
  | x == 0 = False
  | (abs x) < 4 = (mod (abs x) 2) /= 0
  | otherwise = let {
                    n = round $ sqrt $ fromIntegral (abs x)
                    ; xx = abs x
                    ; l = dropWhile (\y -> (mod xx y)>0) [2..n]
                    }
                 in l == []


foo x = dropWhile (\y -> (mod n y)>0) [2..n]
  where n = round $ sqrt $ fromIntegral x


primes :: (Integral a) => [a] -> [a]
primes x = [y | y<-x, isPrime y]

x :: (Integral a, Read a) => a
x = read "123"

solveRPN :: (Num a, Read a) => String -> a  
solveRPN = head . foldl foldingFunction [] . words  
    where   foldingFunction (x:y:ys) "*" = (x * y):ys  
            foldingFunction (x:y:ys) "+" = (x + y):ys  
            foldingFunction (x:y:ys) "-" = (y - x):ys  
            foldingFunction xs numberString = read numberString:xs

main = do
  args <- getArgs
  --mapM putStrLn args
  --print "fuck off"
  let x = read $ (head args)::Int
  --let z = read $ (head args)::(Read a, Num a) => a
  --print x
  --print z
  --mapM print args
  --print args
  let mylist = primes [1..x]
  --print mylist
  print $ last mylist

