import React, { useState, useEffect } from 'react';
import BookFeedback from '../../components/BookFeedback/BookFeedback';
import '../../styles/design-system.css';
import styles from './BookSuggestions.module.css';
import { useUser } from '../../contexts/UserContext';
import UserApi from '../../api/UserApi';

const BookSuggestions = () => {
  const [recommendedBooks, setRecommendedBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { user } = useUser();

  const defaultRecommendedBooks = [
    {
      "bookId": 84,
      "title": "Mystic River",
      "author": "Brian Helgeland",
      "publisher": "Default Publisher",
      "publicationYear": 2002,
      "isbn": "ISBN586400",
      "genre": "Motion picture plays",
      "descriptions": "Unmarked typescript, dated FINAL DRAFT Rev. 09/16/02 (Pink) Clint Eastwood directed, produced, and scored this mystery drama starring Sean Penn, Tim Robbins, and Kevin Bacon. It as released by Warner Brothers Oct. 3, 2003.",
      "coverImageUrl": "https://m.media-amazon.com/images/M/MV5BMTIzNDUyMjA4MV5BMl5BanBnXkFtZTYwNDc4ODM3._V1_.jpg",
      "quantity": 20,
      "availableQuantity": 2,
      "createdAt": "2024-12-30T15:27:37",
      "previewLink": "https://books.google.com.vn/books?id=WpjzzwEACAAJ&lpg=PP1&dq=intitle:Mystic+River&hl=vi&pg=PP1&output=embed"
    },
    {
      "bookId": 85,
      "title": "Sphere",
      "author": "Michael Crichton",
      "publisher": "Default Publisher",
      "publicationYear": 2011,
      "isbn": "ISBN624731",
      "genre": "Life on other planets",
      "descriptions": "No description available.",
      "coverImageUrl": "https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1660273071i/455373.jpg",
      "quantity": 8,
      "availableQuantity": 1,
      "createdAt": "2024-12-30T15:27:37",
      "previewLink": "https://books.google.com.vn/books?id=Ad6CzQEACAAJ&lpg=PP1&dq=intitle:Sphere&hl=vi&pg=PP1&output=embed"
    },
    {
      "bookId": 86,
      "title": "The Pelican Brief",
      "author": "Unknown Author",
      "publisher": "Default Publisher",
      "publicationYear": 2012,
      "isbn": "ISBN831323",
      "genre": "History",
      "descriptions": "No description available.",
      "coverImageUrl": "https://m.media-amazon.com/images/M/MV5BZjA2NmE4MjEtOTkxYy00YjhkLWI2YjgtODFmMGY0Zjc3YTdhXkEyXkFqcGc@._V1_.jpg",
      "quantity": 2,
      "availableQuantity": 16,
      "createdAt": "2024-12-30T15:27:37",
      "previewLink": "https://books.google.com.vn/books?id=-on-zwEACAAJ&lpg=PP1&dq=intitle:The+Pelican+Brief&hl=vi&pg=PP1&output=embed"
    },
  ];

  useEffect(() => {
    if (user?.userId) {
      fetchRecommendedBooks();
    } else {
      setLoading(false);
    }
  }, [user]);

  const fetchRecommendedBooks = async () => {
    try {
      setLoading(true);
      setError(null);

      if (!user || !user.userId) {
        console.error("User ID không hợp lệ");
        setRecommendedBooks(defaultRecommendedBooks);
        return;
      }

      const response = await UserApi.getRecommendedBooks(user.userId);
      const recommendedBooksData = response.data.recommendations;

      if (!recommendedBooksData || !Array.isArray(recommendedBooksData)) {
        console.error("Dữ liệu recommendedBooks không đúng định dạng");
        setRecommendedBooks(defaultRecommendedBooks);
        return;
      }

      const recommendedBooks = recommendedBooksData.map((book, index) => ({
        bookId: book.isbn || index,
        title: book.title,
        author: book.author,
        coverImageUrl: book.cover,
        availableQuantity: 5,
      }));

      setRecommendedBooks(recommendedBooks);
    } catch (error) {
      console.error("Lỗi khi lấy danh sách sách đề xuất:", error);
      setError("Không thể tải danh sách gợi ý. Hiển thị sách mặc định.");
      setRecommendedBooks(defaultRecommendedBooks);
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshRecommendations = () => {
    if (user?.userId) {
      fetchRecommendedBooks();
    }
  };

  if (!user) {
    return (
      <div className={styles.container}>
        <div className={styles.loginPrompt}>
          <div className={styles.loginIcon}>🔐</div>
          <h2>Đăng nhập để xem gợi ý sách</h2>
          <p>Vui lòng đăng nhập để nhận được những gợi ý sách phù hợp với sở thích của bạn.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header Section */}
      <div className={styles.headerSection}>
        <div className={styles.titleContainer}>
          <h1 className={styles.pageTitle}>
            <span className={styles.titleIcon}>🎯</span>
            Gợi Ý Sách Dành Cho Bạn
          </h1>
          <p className={styles.pageSubtitle}>
            Khám phá những cuốn sách được chọn lọc đặc biệt dành riêng cho sở thích của bạn
          </p>
        </div>
        
        <div className={styles.actionButtons}>
          <button 
            onClick={handleRefreshRecommendations}
            className={styles.refreshButton}
            disabled={loading}
          >
            <span className={styles.buttonIcon}>🔄</span>
            {loading ? 'Đang tải...' : 'Làm mới gợi ý'}
          </button>
        </div>
      </div>

      {/* Stats Section */}
      <div className={styles.statsSection}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>📚</div>
          <div className={styles.statContent}>
            <div className={styles.statNumber}>{recommendedBooks.length}</div>
            <div className={styles.statLabel}>Sách được gợi ý</div>
          </div>
        </div>
        
        <div className={styles.statCard}>
          <div className={styles.statIcon}>⭐</div>
          <div className={styles.statContent}>
            <div className={styles.statNumber}>Cá nhân hóa</div>
            <div className={styles.statLabel}>Dựa trên sở thích</div>
          </div>
        </div>
        
        <div className={styles.statCard}>
          <div className={styles.statIcon}>🎯</div>
          <div className={styles.statContent}>
            <div className={styles.statNumber}>AI</div>
            <div className={styles.statLabel}>Thuật toán thông minh</div>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className={styles.errorMessage}>
          <span className={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}

      {/* Loading State */}
      {loading ? (
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner}></div>
          <p className={styles.loadingText}>Đang tạo gợi ý sách cho bạn...</p>
        </div>
      ) : (
        <>
          {/* Instructions Section */}
          <div className={styles.instructionsSection}>
            <h3 className={styles.instructionsTitle}>
              <span className={styles.instructionsIcon}>💡</span>
              Cách sử dụng hệ thống gợi ý
            </h3>
            <div className={styles.instructionsList}>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>1️⃣</span>
                <span>Xem danh sách sách được gợi ý dựa trên sở thích của bạn</span>
              </div>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>2️⃣</span>
                <span>Đánh giá sách bằng cách chọn số sao (1-5 sao)</span>
              </div>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>3️⃣</span>
                <span>Thêm sách yêu thích vào danh sách của bạn</span>
              </div>
              <div className={styles.instructionItem}>
                <span className={styles.instructionIcon}>4️⃣</span>
                <span>Hệ thống sẽ học từ phản hồi để cải thiện gợi ý</span>
              </div>
            </div>
          </div>

          {/* Books Section */}
          <div className={styles.booksSection}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>
                <span className={styles.sectionIcon}>📖</span>
                Danh sách gợi ý ({recommendedBooks.length} cuốn sách)
              </h2>
              <p className={styles.sectionDescription}>
                Những cuốn sách này được chọn lọc dựa trên lịch sử đọc và sở thích của bạn
              </p>
            </div>
            
            <div className={styles.booksContainer}>
              <BookFeedback 
                books={recommendedBooks} 
                userId={user?.userId}
              />
            </div>
          </div>

          {/* Tips Section */}
          <div className={styles.tipsSection}>
            <h3 className={styles.tipsTitle}>
              <span className={styles.tipsIcon}>💭</span>
              Mẹo để nhận được gợi ý tốt hơn
            </h3>
            <div className={styles.tipsList}>
              <div className={styles.tipItem}>
                <span className={styles.tipIcon}>✨</span>
                <span>Đánh giá nhiều sách để hệ thống hiểu rõ sở thích của bạn</span>
              </div>
              <div className={styles.tipItem}>
                <span className={styles.tipIcon}>📚</span>
                <span>Thêm sách vào danh sách yêu thích để cải thiện thuật toán</span>
              </div>
              <div className={styles.tipItem}>
                <span className={styles.tipIcon}>🔄</span>
                <span>Làm mới gợi ý thường xuyên để khám phá sách mới</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default BookSuggestions; 