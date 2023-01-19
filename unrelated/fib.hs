import qualified System.IO as IO 
import qualified System.Environment as E
import Data.String
import Data.Typeable

badFib :: Integer -> Integer
badFib 0 = 0
badFib 1 = 1
badFib n = (badFib (n - 1)) + (badFib (n-2))

fibAkk 0 a b = (a, b, 0)
fibAkk 1 a b = (a, b, 0)
fibAkk n a b = fibAkk (n-1) (a+b) a

isPrime :: Integer -> Bool
isPrime 0 = False
isPrime 1 = True
isPrime 2 = True
isPrime 3 = True
isPrime 4 = False
isPrime x = let { n = round $ sqrt $ fromIntegral x
                ; l = [(mod x i) | i <- [2..n] ]
                ; ll = dropWhile (\k -> k>0) l
                }
             in ll == []

findPrime :: [Int] -> Int -> Bool
findPrime [] _ = True
findPrime l n = (mod n (head l) /= 0) && findPrime (tail l) n 

isPrime' :: Int -> Bool
isPrime' n = findPrime [2..x] n where
  x = round $ sqrt $ fromIntegral n


listPrimes :: Integer -> [Integer]
listPrimes n = [x | x <- [1..n], isPrime(x)]

listPrimes' :: Int -> [Int]
listPrimes' n = [x | x <- [1..n], isPrime'(x)]

main :: IO ()
main = do
  (c : args) <- E.getArgs
  let n = read c :: Integer
  --putStrLn $ show $ (badFib n)
  let (a,b,c) = fibAkk n 1 0
  --putStrLn $ show $ (fibAkk n 1 0)
  --print $ typeOf n
  print a




--
low3a f g h x y = do
  x <- f
  do g <- x
     h y
