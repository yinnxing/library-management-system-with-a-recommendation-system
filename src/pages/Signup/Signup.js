import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import styles from './Signup.module.css';
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext';

const Signup = () => {
    const { login } = useUser();
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();

        // Basic validation
        if (!name || !email || !password || !confirmPassword) {
            setError('Please fill in all fields.');
            return;
        }

        if (password !== confirmPassword) {
            setError('Passwords do not match.');
            return;
        }

        try {
            const response = await UserApi.register(name, email, password);
            const data = response.data.result;

            if (data.success) {
                // Auto login after successful registration
                const loginResponse = await UserApi.login(email, password);
                const loginData = loginResponse.data.result;
                
                if (loginData.authenticated) {
                    localStorage.setItem('accessToken', loginData.accessToken);
                    login({
                        userId: loginData.userId,
                        email: email,
                        role: 'USER'
                    }, loginData.accessToken);
                    
                    navigate('/');
                }
            } else {
                setError(data.message || 'Registration failed.');
            }
        } catch (error) {
            setError('Error connecting to the server. Please try again later.');
        }
    };

    return (
        <div className={styles.signupContainer}>
            <h2>Create Account</h2>
            {error && <p className={styles.error}>{error}</p>}
            <form onSubmit={handleSubmit}>
                <div className={styles.formGroup}>
                    <label htmlFor="name">Full Name</label>
                    <input
                        type="text"
                        id="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Enter your full name"
                    />
                </div>

                <div className={styles.formGroup}>
                    <label htmlFor="email">Email</label>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="Enter your email"
                    />
                </div>

                <div className={styles.formGroup}>
                    <label htmlFor="password">Password</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                    />
                </div>

                <div className={styles.formGroup}>
                    <label htmlFor="confirmPassword">Confirm Password</label>
                    <input
                        type="password"
                        id="confirmPassword"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        placeholder="Confirm your password"
                    />
                </div>

                <button type="submit" className={styles.signupButton}>Create Account</button>
                
                <div className={styles.loginLink}>
                    Already have an account? <Link to="/login">Login</Link>
                </div>
            </form>
        </div>
    );
};

export default Signup; 