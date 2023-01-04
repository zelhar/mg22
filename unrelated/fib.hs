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
