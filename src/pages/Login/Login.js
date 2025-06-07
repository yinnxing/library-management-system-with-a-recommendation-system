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
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate(); 

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        if (!email || !password) {
            setError('Vui lòng điền đầy đủ email và mật khẩu.');
            return;
        }

        setIsLoading(true);

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
                setError(data.message || 'Thông tin đăng nhập không hợp lệ.');
            }
        } catch (error) {
            setError('Lỗi kết nối đến máy chủ. Vui lòng thử lại sau.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleGoogleLogin = async () => {
        try {
            // Redirect to Google OAuth with the correct callback URL
            window.location.href = `${UserApi.API_BASE_URL}/oauth2/authorization/google?redirect_uri=${window.location.origin}/oauth2/callback`;
        } catch (error) {
            setError('Lỗi kết nối đến Google authentication.');
        }
    };
   
    return (
        <div className={styles.loginPage}>
        <div className={styles.loginContainer}> 
                <div className={styles.loginHeader}>
            
                    <h2 className={styles.loginTitle}>Đăng Nhập</h2>
                    <p className={styles.loginSubtitle}>
                        Chào mừng bạn trở lại! Vui lòng đăng nhập để tiếp tục.
                    </p>
                </div>

                <div className={styles.loginForm}>
                    {error && (
                        <div className={styles.errorMessage}>
                            <span className={styles.errorIcon}>⚠️</span>
                            {error}
                        </div>
                    )}

            <form onSubmit={handleSubmit}>
                <div className={styles.formGroup}> 
                            <label htmlFor="email" className={styles.formLabel}>
                                Email
                            </label>
                            <div className={styles.inputContainer}>
                                <span className={styles.inputIcon}>✉️</span>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                                    placeholder="Nhập địa chỉ email của bạn"
                                    className={styles.formInput}
                                    disabled={isLoading}
                    />
                            </div>
                </div>

                <div className={styles.formGroup}> 
                            <label htmlFor="password" className={styles.formLabel}>
                                Mật khẩu
                            </label>
                            <div className={styles.inputContainer}>
                                <span className={styles.inputIcon}>🔒</span>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Nhập mật khẩu của bạn"
                                    className={styles.formInput}
                                    disabled={isLoading}
                    />
                </div>
                        </div>

                        <button 
                            type="submit" 
                            className={styles.loginButton}
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <div className={styles.spinner}></div>
                                    Đang đăng nhập...
                                </>
                            ) : (
                                <>
                                    <span className={styles.buttonIcon}>🚀</span>
                                    Đăng Nhập
                                </>
                            )}
                        </button>
            </form>
            
                    <div className={styles.divider}>
                        <span className={styles.dividerText}>Hoặc</span>
                    </div>

                    <button 
                        className={styles.googleButton}
                        onClick={handleGoogleLogin}
                        disabled={isLoading}
                    >
                        <img 
                            src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/1200px-Google_%22G%22_logo.svg.png" 
                            alt="Google" 
                            className={styles.googleIcon}
                        />
                        Đăng nhập với Google
                    </button>

                    <div className={styles.signupPrompt}>
                        <p>
                            Chưa có tài khoản? 
                            <Link to="/signup" className={styles.signupLink}>
                                Đăng ký ngay
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
