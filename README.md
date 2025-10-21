# P&L MPC Historian Dashboard Tool

## Overview
This is the first draft of a Dash-based web application for visualizing and analyzing MPC historian data. Currently includes basic demo functionality with plans for significant expansion.

## Current Project Structure
- **app.py**: Initializes the Dash application and sets up the main layout
- **ids.py**: Central repository of ID constants used throughout the application
- **assets/**: Styling and client-side JavaScript
  - **custom-styles.css**: Main styling for components
  - **base-styles.css**: Base layout styling
  - **viewport_helpers.js**: Improves dropdown and popover positioning
- **model_doc/**: Markdown-based model documentation editor
- **data_input/**: Data import and visualization components
  - **database/**: SQLite database loading and table preview
  - **plots/**: Plot creation and configuration
    - **time_series/**: Time series plot implementation
    - **mpc/**: MPC visualization implementation
- **dashboards/**: Dashboard layout components
- **export/**: Data export functionality (placeholder)

## Upcoming Features
- Support for much more plot types beyond Time Series and MPC
- Draggable dashboard capabilities for customized layouts
- Export functionality (pdf, video, ...)
- User management and authentication
- Persistence and sharing options

## Setup
1. Navigate to project directory: `cd historian-dash-app`
2. Install dependencies: `pip install -r requirements.txt`
3. Run application: `python app.py`
