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
