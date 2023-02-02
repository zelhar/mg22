import Control.Applicative
import Control.Monad
import Control.Monad.State.Lazy
import Data.List
import System.Environment
import System.IO
import Data.Tuple
import Data.Typeable

ticker :: Int -> Int
ticker = (+1)

tick :: State Int Int
tick = do n <- get
          put (n+1)
          return n

tack :: State Int Int
tack = get >>= (\n -> put (n+1) >> return n)

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

mnumerator' :: [a] -> State Int [Int]
mnumerator' [] = return []
mnumerator' (x:xs) = tick >>= (\n -> 
                              (mnumerator' xs) >>= 
                              (\ns -> return (n : ns)))

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


-----------------------------------------------------

newtype ZT s a = Z (s -> (a,s)) 

app :: ZT s a -> s -> (a,s)
app (Z f) = f

instance Functor (ZT s) where
  --fmap :: (a -> b) -> ZT s a -> ZT s b
  fmap f st = Z (\s -> let (x,s') = app st s in (f x, s'))

instance Applicative (ZT s) where
  --pure :: a -> ZT s a
  pure x = Z (\s -> (x,s))
  -- (<*>) :: ZT s (a -> b) -> ZT s a -> ZT s b
  stf <*> sta = Z (\s -> 
                  let (f, s') = app stf s
                      (a, s'') = app sta s' in (f a, s''))

instance Monad (ZT s) where
  --(>>=) :: ZT s a -> (a -> ZT s b) -> ZT s b
  st >>= f = Z (\s -> 
               let (a,s') = app st s in app (f a) s')

tickz :: ZT Int Int
tickz = Z (\n -> (n,n+1))

znum :: [a] -> ZT Int [Int]
znum [] = return []
znum (x:xs) = do n <- tickz
                 ns <- znum xs
                 return (n:ns)

znum' :: [a] -> ZT Int [Int]
znum' [] = return []
znum' (x:xs) = tickz >>= \n -> ((znum xs) >>= (\ns -> return (n : ns)) )


-- notice how bind >>= always pass the result (\a ->) and the application
-- passes the state S (\s ->)... this is why tick and enumarate etc. work like
-- they do. This is why n <- ... implies n has the result type (neccessitated
-- from the way >>= is defind on states and from the fact that State Int is a
-- monad on a ....)


































































