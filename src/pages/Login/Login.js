import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom'; 
import styles from './Login.module.css'; 
import UserApi from '../../api/UserApi';
import Cookies from 'js-cookie';
import { useUser } from '../../contexts/UserContext'; 
import { jwtDecode } from 'jwt-decode'; 
import { useEffect } from 'react';

const Login = () => {
    const { login } = useUser();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [showLoginModal, setShowLoginModal] = useState(false);
    const navigate = useNavigate(); 

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!email || !password) {
            setError('Please fill in both email and password.');
            return;
        }

        try {
            const response = await UserApi.login(email, password);
            const data = response.data.result;
            console.log(data);

            if (data.authenticated) {
                localStorage.setItem('accessToken', data.accessToken);
                Cookies.set('refreshToken', data.refreshToken, {
                    secure: false,
                });
                const decodedToken = jwtDecode(data.accessToken);
                console.log("DECODED:", decodedToken)
                const userData = {
                    userId: decodedToken.userId,
                    email: decodedToken.sub, 
                    role: decodedToken.role
                };

                login(userData, data.accessToken);

                const isAdmin = decodedToken.role === 'ADMIN';
                if (isAdmin) {
                    navigate('/admin');  
                } else {
                    navigate('/'); 
                }

            } else {
                setError(data.message || 'Invalid login credentials.');
            }
        } catch (error) {
            setError('Error connecting to the server. Please try again later.');
        }
    };

    const handleGoogleLogin = async () => {
        try {
            // Redirect to Google OAuth with the correct callback URL
            window.location.href = `${UserApi.API_BASE_URL}/oauth2/authorization/google?redirect_uri=${window.location.origin}/oauth2/callback`;
        } catch (error) {
            setError('Error connecting to Google authentication.');
        }
    };

    const handleFacebookLogin = async () => {
        try {
            // This would typically redirect to Facebook OAuth
            window.location.href = `${UserApi.API_BASE_URL}/auth/facebook`;
        } catch (error) {
            setError('Error connecting to Facebook authentication.');
        }
    };
   
    return (
        <div className={styles.loginContainer}> 
            <h2>Login</h2>
            {error && <p className={styles.error}>{error}</p>} 
            <form onSubmit={handleSubmit}>
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

                <button type="submit" className={styles.loginButton}>Login</button>
                
                <div className={styles.signupLink}>
                    Don't have an account? <Link to="/signup">Sign up</Link>
                </div>
            </form>
            
            <div className={styles.socialLogin}>
                <p>Or login with</p>
                <div className={styles.socialButtons}>
                    <button 
                        className={styles.googleButton}
                        onClick={handleGoogleLogin}
                    >
                        <img 
                            src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" 
                            alt="Google" 
                        />
                        Google
                    </button>
                    <button 
                        className={styles.facebookButton}
                        onClick={handleFacebookLogin}
                    >
                        <img 
                            src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Facebook_Logo_%282019%29.png/1024px-Facebook_Logo_%282019%29.png" 
                            alt="Facebook" 
                        />
                        Facebook
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Login;
