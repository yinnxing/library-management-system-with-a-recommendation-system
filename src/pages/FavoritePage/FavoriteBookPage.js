import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import UserApi from '../../api/UserApi';
import { useUser } from '../../contexts/UserContext'; 
import BookList from '../../components/BookList/BookList';
import '../../styles/design-system.css';
import styles from './FavoriteBookPage.module.css';

const FavoriteBookPage = () => {
  const { user } = useUser();
  const [favoriteBooks, setFavoriteBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('grid');
  const [sortBy, setSortBy] = useState('dateAdded');
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
  const fetchFavoriteBooks = async () => {
      if (!user?.userId) return;
      
      setLoading(true);
      setError(null);
      
    try {
      const response = await UserApi.getFavoriteBooks(user.userId);
      console.log('Response từ API:', response);

      if (response.data.code === 0) {
        setFavoriteBooks(response.data.result);
      } else {
          setError(response.data.message || 'Không thể tải danh sách yêu thích');
      }
    } catch (error) {
      console.error('Lỗi gọi API:', error); 
        setError('Có lỗi xảy ra khi tải danh sách yêu thích');
      } finally {
      setLoading(false);
    }
  };

  fetchFavoriteBooks();
  }, [user?.userId]);

  const filteredAndSortedBooks = favoriteBooks
    .filter(book => 
      book.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      book.author.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'title':
          return a.title.localeCompare(b.title);
        case 'author':
          return a.author.localeCompare(b.author);
        case 'year':
          return b.publicationYear - a.publicationYear;
        default:
          return 0;
      }
    });

  const handleRefresh = () => {
    const fetchFavoriteBooks = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await UserApi.getFavoriteBooks(user.userId);
        if (response.data.code === 0) {
          setFavoriteBooks(response.data.result);
        } else {
          setError(response.data.message || 'Không thể tải danh sách yêu thích');
        }
      } catch (error) {
        setError('Có lỗi xảy ra khi tải danh sách yêu thích');
      } finally {
        setLoading(false);
      }
    };

    fetchFavoriteBooks();
  };

  // Callback to refresh favorites when BookList updates them
  const handleFavoriteUpdate = () => {
    // Refresh the favorites list after a short delay to ensure API update is complete
    setTimeout(() => {
      handleRefresh();
    }, 500);
  };

  return (
    <div className={styles.container}>
      {/* Header Section */}
      <div className={styles.headerSection}>
        <div className={styles.titleContainer}>
          <h1 className={styles.pageTitle}>
            <span className={styles.titleIcon}>❤️</span>
            Sách Yêu Thích
          </h1>
          <p className={styles.pageSubtitle}>
            Bộ sưu tập những cuốn sách bạn yêu thích nhất
          </p>
        </div>
        
        <div className={styles.headerActions}>
          <button 
            className={styles.refreshButton}
            onClick={handleRefresh}
            disabled={loading}
          >
            <span className={styles.buttonIcon}>🔄</span>
            Làm mới
          </button>
        </div>
      </div>

      {/* Stats Section */}
      <div className={styles.statsSection}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📚</div>
          <div className={styles.statContent}>
            <div className={styles.statNumber}>{favoriteBooks.length}</div>
            <div className={styles.statLabel}>Sách yêu thích</div>
          </div>
        </div>
        
        <div className={styles.statCard}>
          <div className={styles.statIcon}>🔍</div>
          <div className={styles.statContent}>
            <div className={styles.statNumber}>{filteredAndSortedBooks.length}</div>
            <div className={styles.statLabel}>Kết quả hiển thị</div>
          </div>
        </div>
      </div>

      {/* Controls Section */}
      {favoriteBooks.length > 0 && (
        <div className={styles.controlsSection}>
          <div className={styles.searchContainer}>
            <input
              type="text"
              placeholder="Tìm kiếm sách yêu thích..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={styles.searchInput}
            />
            <span className={styles.searchIcon}>🔍</span>
          </div>

          <div className={styles.controlsRight}>
            <div className={styles.sortContainer}>
              <label className={styles.sortLabel}>Sắp xếp:</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className={styles.sortSelect}
              >
                <option value="dateAdded">Mới thêm</option>
                <option value="title">Tên sách A-Z</option>
                <option value="author">Tác giả A-Z</option>
                <option value="year">Năm xuất bản</option>
              </select>
            </div>

            <div className={styles.viewToggle}>
              <button
                className={`${styles.viewButton} ${viewMode === 'grid' ? styles.active : ''}`}
                onClick={() => setViewMode('grid')}
              >
                ⊞
              </button>
              <button
                className={`${styles.viewButton} ${viewMode === 'list' ? styles.active : ''}`}
                onClick={() => setViewMode('list')}
              >
                ☰
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className={styles.errorMessage}>
          <span className={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}

      {/* Content Section */}
      <div className={styles.contentSection}>
        {loading ? (
          <div className={styles.loadingContainer}>
            <div className={styles.loadingSpinner}></div>
            <p className={styles.loadingText}>Đang tải danh sách yêu thích...</p>
          </div>
        ) : filteredAndSortedBooks.length > 0 ? (
          <BookList books={filteredAndSortedBooks} userId={user?.userId} onFavoriteUpdate={handleFavoriteUpdate} />
        ) : favoriteBooks.length === 0 ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>💔</div>
            <h3 className={styles.emptyTitle}>Chưa có sách yêu thích</h3>
            <p className={styles.emptyDescription}>
              Hãy khám phá thư viện và thêm những cuốn sách bạn yêu thích vào danh sách này
            </p>
            <Link to="/books" className={styles.exploreButton}>
              <span className={styles.buttonIcon}>🔍</span>
              Khám phá sách
            </Link>
          </div>
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>🔍</div>
            <h3 className={styles.emptyTitle}>Không tìm thấy sách nào</h3>
            <p className={styles.emptyDescription}>
              Thử thay đổi từ khóa tìm kiếm hoặc bộ lọc
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FavoriteBookPage;
