import React from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/poppins/400.css';
import '@fontsource/poppins/500.css';
import '@fontsource/poppins/600.css';
import '@fontsource/poppins/700.css';
import Settings from './Settings';
import './styles.css';
import './settings.css';

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<Settings />);
