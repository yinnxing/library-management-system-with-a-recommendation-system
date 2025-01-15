import React, { useEffect, useState } from 'react';
import { Link, useNavigate, NavLink } from 'react-router-dom';
import Search from '../Search/Search';
import styles from './Header.module.css';
import { useUser } from '../../contexts/UserContext'; 
import Cookies from 'js-cookie';
import HeadlessTippy from '@tippyjs/react/headless';



const Header = () => {
  const { user, login, logout } = useUser();
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false); 

  useEffect(() => {
  const token = localStorage.getItem('accessToken');
  if (token && !isAuthenticated) { 
    console.log(user);
    setIsAuthenticated(true);
  }
  
}, [isAuthenticated, login, user]);



// useEffect(() => {
//   if (isAuthenticated) {
//     setVisible(false); 
//   }
// }, [isAuthenticated]);

  const handleLogout = () => {
    logout(); // Đăng xuất người dùng và xóa thông tin khỏi Context
    localStorage.removeItem('accessToken'); // Xóa accessToken khỏi localStorage
    Cookies.remove('refreshToken'); // Xóa refreshToken khỏi cookie
    navigate('/'); 
    window.location.reload()
  };

  const [visible, setVisible] = useState(false); // State kiểm soát tooltip

 return (
  <header className={styles.header}>
    {/* Logo */}
    <div className={styles.logo}>
      <Link to="/">Thư Viện</Link>
    </div>

    {/* Thanh Tìm Kiếm */}
    <div className={styles.searchWrapper}>
      <Search /> {/* Component Search sẽ xử lý logic tìm kiếm và hiển thị kết quả */}
    </div>

    {/* Menu Điều Hướng */}
    <nav className={styles.nav}>
      <NavLink
    to="/favorite"
    className={({ isActive }) => isActive ? styles.activeLink : styles.button}
  >
    Danh sách yêu thích
  </NavLink>
  <NavLink
    to="/borrowBook"
    className={({ isActive }) => isActive ? styles.activeLink : styles.button}
  >
    Mượn sách
  </NavLink>


      {/* Kiểm tra xem người dùng đã đăng nhập chưa */}
      {!isAuthenticated ? (
        <Link to="/login" className={styles.loginLink}>Đăng nhập/Đăng ký</Link>
      ) : (
        <HeadlessTippy
          visible={visible} 
          onClickOutside={() => setVisible(false)} 
          placement="bottom"
          interactive={true} 
          arrow={false} 
          render={(attrs) => (
            <div
              className={styles.userProfileMenu}
              tabIndex="-1"
              {...attrs} 
            >
              <div className={styles.userProfile}>
                <button>
                  <Link to="/user-profile">{user?.email}</Link>
                </button>
                <button onClick={handleLogout}>Đăng xuất</button>
              </div>
            </div>
          )}
        >
          <div
            className={styles.userMenu}
            onClick={() => setVisible(!visible)} 
          >
            <span className={styles.username}>Tài khoản</span>
            <img
              src="https://media.istockphoto.com/id/1443209389/vi/vec-to/s%C3%A1ch-v%E1%BA%BD-ngu%E1%BB%87ch-ngo%E1%BA%A1c.jpg?s=612x612&w=0&k=20&c=a-bJw6Ic1FLS5Ri4TlRqOh0bREGyVO_2W52seuNupuk=" 
              alt="avatar"
              className={styles.avatar}
            />
          </div>
        </HeadlessTippy>
      )}
    </nav>
  </header>
);
}

export default Header;
