library(dplyr)
library(shiny)
library(bslib)

ui <- page_navbar(
  title = "Germany's development dashboard",
  bg = "#2D89C8",
  inverse = TRUE,
  nav_panel(title = "Landing", ),
  nav_panel(title = "About", p("This will describe the project"))
)

server <- function(input, output) {}

shinyApp(ui = ui, server = server)
