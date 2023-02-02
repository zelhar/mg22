import Control.Applicative
import Control.Monad
import Control.Monad.State.Lazy
import Data.List
import System.Environment
import System.IO
import Data.Tuple

tick :: State Int Int
tick = do n <- get
          put (n+1)
          return n

tock :: State Int Int
tock = state (\s -> (s,s+1))

data Tree a
  = Leaf a
  | Node (Tree a) (Tree a)
  deriving (Show)

tree :: Tree Char
tree = Node (Node (Leaf 'a') (Leaf 'b')) (Leaf 'c')

numerator :: [a] -> Int -> ([Int], Int)
numerator [] n = ([], n)
numerator (x:xs) n = (n : ys, m) 
  where (ys,m) = numerator xs (n+1)


mnumerator :: [a] -> State Int [Int]
mnumerator [] = return []
mnumerator (x:xs) = do n <- tick
                       ys <- mnumerator xs
                       return (n : ys)


foo :: [Int] -> [Int]
foo xs = do x <- xs
            [x+1]
            0:[x+2]
            
zzz :: State Int String
zzz = 
  do n <- get
     put (n+1)
     return (show n)


www :: State Int String
www = state (\s -> (show s,s+1))

putty :: s -> State s ()
putty s = state (const ((), s))

moo :: Int -> State Int Int
moo s = put s >> tick --when run returns (s,s+1) on any input state

--appS :: State s a -> s -> State a ()
appS xs ss = do x <- xs
                let y = (x ss)
                return $ fst y

strangeid :: State s a -> State s a
strangeid xs = do x <- xs
                  return x
