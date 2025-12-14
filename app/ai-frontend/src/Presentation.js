import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import NeoDataLogo from './assets/neodata.jpg';

const Presentation = () => {
  const [currentSlide, setCurrentSlide] = useState(0);

  const KFK_LOGO_URL = "https://www.kfk.hr/static/uploads/2020/10/logo-light.png"
  const COMMINUS_LOGO_URL = "https://www.comminus.hr/wp-content/uploads/2025/06/logo_400x200.png"

  const slides = [
    {
      title: 'Tim FERSADA - naše rješenje:',
      content: 'Bok! Za Vas smo pripremili dva rješenja!',
      bgColor: '#182443ff',
    },
    {
      title: 'I. rješenje - "Brzo rješenje"',
      content: 'Možete vidjeti i isprobati ovaj live demo.',
      bgColor: '#c4204bff',
    },
    {
      title: 'II. rješenje - "Industrijsko rješenje"',
      content: 'Za to ćete morati poslušati naš pitch!',
      bgColor: '#161d7bff',
    },
    {
      title: 'Čekamo Vas!',
      content: 'Jedva čekamo pokazati vam što imamo!',
      bgColor: '#08196fff',
    },
  ];

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const goToSlide = (index) => {
    setCurrentSlide(index);
  };

  const slide = slides[currentSlide];

  return (
    <div className="app-shell">
      <nav className="top-bar">
        <div className="brand">
          <div className="brand-stack">
            <img src={NeoDataLogo} alt="NeoData" className="brand-logo neo" />
          </div>
          <span className="brand-separator" aria-hidden="true" />
          <div className="brand-stack">
            <img src={KFK_LOGO_URL} alt="KFK" className="brand-logo kfk" />
            <small>KFK QC Challenge - NeoData</small>
          </div>
        </div>
        <div className="top-actions">
          <div className="partner-logo">
            <img src={COMMINUS_LOGO_URL} alt="Comminus" />
          </div>
          <Link className="ghost secondary" to="/">
            Back to App
          </Link>
          <a className="ghost secondary" href="https://www.kfk.hr" target="_blank" rel="noreferrer">
            KFK Portal
          </a>
        </div>
      </nav>

      <main className="content" style={{ padding: '2rem' }}>
      {/* Slide Display */}
      <div
        style={{
          backgroundColor: slide.bgColor,
          padding: '4rem 2rem',
          borderRadius: '8px',
          textAlign: 'center',
          minHeight: '500px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          marginBottom: '2rem',
        }}
      >
        <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>{slide.title}</h1>
        <p style={{ fontSize: '1.2rem', lineHeight: '1.6', maxWidth: '600px' }}>
          {slide.content}
        </p>
      </div>

      {/* Navigation Buttons */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginBottom: '2rem' }}>
        <button
          onClick={prevSlide}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          ← Previous
        </button>
        <button
          onClick={nextSlide}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Next →
        </button>
      </div>

      {/* Slide Indicators */}
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
        {slides.map((_, index) => (
          <button
            key={index}
            onClick={() => goToSlide(index)}
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: currentSlide === index ? '#007bff' : '#ccc',
              border: 'none',
              cursor: 'pointer',
            }}
          />
        ))}
      </div>

      {/* Slide Counter */}
      <div style={{ textAlign: 'center', marginTop: '2rem', color: '#666' }}>
        Slide {currentSlide + 1} of {slides.length}
      </div>
      </main>
    </div>
  );
};

export default Presentation;
