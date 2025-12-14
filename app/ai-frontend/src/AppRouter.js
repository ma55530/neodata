import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import App from './App';
import Presentation from './Presentation';

const AppRouter = () => (
  <Router>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/presentation" element={<Presentation />} />
    </Routes>
  </Router>
);

export default AppRouter;
