import Control.Applicative
import Control.Monad
import Control.Monad.State.Lazy
import Data.List
import System.Environment
import System.IO
import Data.Tuple


--(<$>) :: (a -> b) -> f a -> f b
--t (+) [1] 
--      Num a => [a -> a]

-- (<*>) :: f (a -> b) -> f a -> f b

tack :: State String Int
tack = do n <- get
          put "$"
          return $ length n



tick :: State Int Int
tick = do n <- get
          put (n+1)
          return n

--runState :: State s a -> s -> (a,s)
foo = runState tick 1

-- runState (put 1) 123
-- runState get 1

data Tree a = Leaf a | Node (Tree a) (Tree a)
              deriving Show

tree :: Tree Char
tree = Node (Node (Leaf 'a') (Leaf 'b')) (Leaf 'c')

mlabel :: Tree a -> State Int (Tree Int)
mlabel (Leaf _) = do n <- tick
                     return (Leaf n)
mlabel (Node l r) = do l' <- mlabel l
                       r' <- mlabel r
                       return (Node l' r')

tree' = runState  (mlabel tree) 0


