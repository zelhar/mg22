--import Data.Functor
-- chekc out hackage.haskell.org
import Control.Monad
import Control.Applicative
import Language.Haskell.TH (thisModule)
import Data.Typeable
import Data.Data



{-
## Functor
type f Functor comes with a function
fmap :: (a -> b) -> f a -> f b
which adheres to the two laws of Functors:
fmap id == id; fmap (f . g) == fmap f . fmap g
<$> == infix fmap
(<$) :: a -> f b -> f a -- fmap. const
-}

la = [1..10]
lb = fmap fromIntegral la
lc = fmap sqrt lb
ld = fmap (sqrt . fromIntegral) la

-- example with Maybe
--ma =  fmap show $ Just 11
ma = show <$> Just 11
mb = 1 <$ [1..10]


{- 
## Applicative
class Functor f => Applicative f 
which must implement pure and <*> or loftA2
(<*>) == liftA2 id
liftA2 f x y = f <$> x <*> y
pure :: a -> f a
(<*>) :: f (a -> b) -> f a -> f b
lisftA2 :: (a -> b -> c) -> f a -> f b -> f c
(*>) :: f a -> f b -> fb
(<*) :: f a -> f b -> fa

identity law:
pure id <*> v = v
composition law:
pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
homnomorphism law:
pure f <*> pure x = pure (f x)
interchange law:
u <*> pure y = pure ($ y) <*> u

If f is also a Monad then it must hold
pure = return
(*>) = (>>)
m1 <*> m2 = m1 >>= (\x1 -> m2 >>= (\x2 -> return (x1 x2)))


## Alternative
class Applicative f => Alternative f where
a monoid on applicative functorsv
empty :: f a
(<|>) :: f a -> f a -> f a
some :: f a -> f [a] --one or more
many :: f a -> f [a] --zero or more

if defines, some and many should be the least soloutions for
some v = (:) <$> v <*> many v
many v = some v <|> pure []
-}

x :: Maybe Int
x = pure 42 --same pure == Just

foo x y = x + y
bar = foo <$> [1..4]
res = bar <*> [1]

{-
## Monad
class Applicative m => Monad m where...
Monad laws:
Left identity:
return a >>= k == k a
Right identity:
m >>= return == m
Associativity:
m >>= (\x -> k x >>= h) == (m >>= k) >>= h
pure == return
m1 <*> m2 == m1 >>= (\x1 -> m1 >>== (\x2 -> return (x1 x2)))
fmap f xs = xs >>= return . f
(>>) = (*>)

return :: a -> m a
(>>=) :: m a -> (a -> m b) -> m b
(>>=) as bs == do {a <-as; bs a;}
(>>) :: m a -> m b -> m b
(>>) as bs == do {as; bs;}

class (Monad m, alternative m) => MonadPlus m where ...
monad that supports choice and failure
mzero :: m a
mzero >>= f = mzero
v >> mzero = mzero
mzero == empty (defauly)
mplus :: m a -> m a -> m a
mplus == (<|>)
-}

test1 ls = ls >>= \x -> [show x]
test1a ls = do {x <- ls; [show x];}

test2 ls = ls >>= \x -> [show x] >>= \x -> [(++ "foo") x]
test2a ls = do { x<- ls
               ; let y = show x
               ; [(++ "foo") y]
               }

divisors :: Integer -> [Integer]
divisors n = [k | k<-[1..l], n `mod` k == 0] 
  where
     l = round $ sqrt $ fromInteger n


alter = (=<<) divisors

alter' :: [Integer] -> [Integer]
alter' ls = do {
               x <- ls;
               divisors x;
               }

fooo :: [Integer] -> Integer -> [Integer] 
fooo xs n = do 
  { x<-xs
  ; if mod n x == 0 then 
                    return x;
                    else do [];
  }



divisors' :: [Integer] -> [[Integer]]
divisors' ns = do {
                  n <- ns;
                  [divisors n];
                  }

isit x =
  if x == 0
    then 0
    else 2

talkit :: IO String
talkit = do {
            print "give me a line";
            getLine
            }

talkit2 :: IO String
talkit2 = do {
             print "give me a line";
             putStrLn "asshole!";
             x <- getLine;
             if x=="" then return x
                        else do {putStrLn x; talkit2;}
             }

--data Tree a = Nil | Node a (Tree a) (Tree a) deriving (Typeable, Data, Show)

--data Tree a = Nil | node parent leftsub rightsub
data Tree a = Nil | Root a (Tree a) (Tree a) (Tree a) deriving (Typeable, Data, Show)


data Foo = NulF | B Bar deriving (Typeable, Data, Show)
data Bar = NulB | F Foo deriving (Typeable, Data, Show)

data BList a = Non | Link a [a] [a] deriving (Show)



root :: Tree a -> Maybe a
root Nil = Nothing
root (Root a _ _ _) = Just a

parent :: Tree a -> Maybe a
parent Nil = Nothing
parent (Root _ p _ _) = root p

lchild :: Tree a -> Maybe a
lchild Nil = Nothing
lchild (Root _ _ l _) = root l

rchild :: Tree a -> Maybe a
rchild Nil = Nothing
rchild (Root _ _ _ r) = root r

killLeft :: Tree a -> Tree a
killLeft Nil = Nil
killLeft (Root x p l r) = Root x p Nil r

killRight :: Tree a -> Tree a
killRight Nil =Nil
killRight (Root x p l r) = Root x p l Nil




main = do {
          print "Functors:";
          print $ show <$> Just 11;
          print $ 1 <$ [0..9];
          print $ show <$> (1 <$ [0..9]);
          print $ show <$> (+1) <$> [0..9];
          print "Applicative:";
          print $ liftA2 (+) [0..3] [0];
          print $ (+) <$> [0..3] <*> [0];
          }
