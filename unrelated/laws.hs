--import Data.Functor

import Control.Monad
import Control.Applicative




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
-}

x :: Maybe Int
x = pure 42 --same pure == Just

foo x y = x + y
bar = foo <$> [1..4]
res = bar <*> [1]

{-
## Monad
}-




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
