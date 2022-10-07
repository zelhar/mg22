# Select default CRAN mirror for package installation.
options(repos=c(CRAN="https://cran.wu.ac.at/"))

# Print maximum 1000 elements.
options(max.print=1000)

# broser
options(browser="/usr/bin/firefox")


# No scientific notation.
options(scipen=10)

# No graphics in menus.
options(menu.graphics=FALSE)

# Auto-completion for package names.
utils::rc.settings(ipck=TRUE)

# Invisible environment to mask defined functions
.env = new.env()

# Quit R without asking to save.
.env$q <- function (save="no", ...) {
      quit(save=save, ...)
}
# Attach the environment to enable functions.
attach(.env, warn.conflicts=FALSE)

