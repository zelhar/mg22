import Control.Monad

-- Functors
-- type f Functor comes with a function fmap :: (a -> b) -> f a -> f b
-- which adheres to the two laws of Functors:
-- fmap id == id; fmap (f . g) == fmap f . fmap g
-- map :: (a -> b) -> [a] -> [b]
-- example with the List functor: fmap = map
-- fmap:: (Functor f) => (a -> b) -> f a -> f b
la = [1..10]
lb = fmap fromIntegral la
lc = fmap sqrt lb
ld = fmap (sqrt . fromIntegral) la
ld2 = fmap sqrt . fmap fromIntegral $ la

-- example with Maybe
ma =  fmap show $ Just 11


-- Applicative

main = do {
          --putStrLn ma
          ; return 1
          }
