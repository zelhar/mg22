import Control.Monad
import Control.Applicative
--import System.IO
import Data.Char
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
mains =
 do name <- getLine
    let loudName = map toUpper name
    putStrLn ("Hello " ++ loudName ++ "!")
    putStrLn ("Oh boy! Am I excited to meet you, " ++ loudName)

mainss = do { name <- getLine
            ; let loudName = map toUpper name
            ; putStrLn ("Hello " ++ loudName ++ "!");
            }
            
