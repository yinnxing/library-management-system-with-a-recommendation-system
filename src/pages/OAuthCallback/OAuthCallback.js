import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useUser } from '../../contexts/UserContext';
import { jwtDecode } from 'jwt-decode';
import styles from './OAuthCallback.module.css';

const OAuthCallback = () => {
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useUser();

  useEffect(() => {
    const processOAuthCallback = async () => {
      try {
        // Parse the URL parameters
        const params = new URLSearchParams(location.search);
        const token = params.get('token');
        
        if (!token) {
          setError('No authentication token received');
          setLoading(false);
          return;
        }

        // Store the token and decode it
        localStorage.setItem('accessToken', token);
        
        try {
          const decodedToken = jwtDecode(token);
          
          // Create user data from token
          const userData = {
            userId: decodedToken.userId,
            email: decodedToken.sub,
            role: decodedToken.role
          };
          
          // Update authentication state
          login(userData, token);
          
          // Redirect based on role
          const isAdmin = decodedToken.role === 'ADMIN';
          if (isAdmin) {
            navigate('/admin');
          } else {
            navigate('/');
          }
        } catch (decodeError) {
          console.error('Error decoding token:', decodeError);
          setError('Invalid authentication token');
          setLoading(false);
        }
      } catch (err) {
        console.error('OAuth callback error:', err);
        setError('Authentication failed. Please try again.');
        setLoading(false);
      }
    };

    processOAuthCallback();
  }, [location.search, login, navigate]);

  if (loading) {
    return (
      <div className={styles.callbackContainer}>
        <div className={styles.loader}></div>
        <p>Completing authentication...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.callbackContainer}>
        <div className={styles.error}>
          <h2>Authentication Error</h2>
          <p>{error}</p>
          <button onClick={() => navigate('/login')} className={styles.returnButton}>
            Return to Login
          </button>
        </div>
      </div>
    );
  }

  return null;
};

export default OAuthCallback;
