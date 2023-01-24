import Control.Monad

-- Functors
-- type f Functor comes with a function fmap :: (a -> b) -> f a -> f b
-- which adheres to the two laws of Functors:
-- fmap id == id; fmap (f . g) == fmap f . fmap g
-- compare with lists: map :: (a -> b) -> [a] -> [b]
-- example with the List functor: fmap = map
-- fmap:: (Functor f) => (a -> b) -> f a -> f b
la = [1..10]
lb = fmap fromIntegral la
lc = fmap sqrt lb
ld = fmap (sqrt . fromIntegral) la
ld2 = fmap sqrt . fmap fromIntegral $ la

-- <$>  == infix fmap
-- note: ($) :: (a -> b) -> a -> b
-- and: (<$>) :: Functor f => (a -> b) -> f a -> f b

-- example with Maybe
ma =  fmap show $ Just 11


-- class Functor f => Applicative f where
-- pure :: a -> f a
-- lift value
-- (<*>) :: f (a -> b) -> f a -> f b
-- lift 2 values
-- liftA2 :: (a -> b -> c) -> f a -> f b -> f c
-- must satisfy:
-- (<*>) = liftA2 id
-- liftA2 f x y = f <$> x <*> y

main = do {
          --putStrLn ma
          ; return 1
          }
